from __future__ import annotations

from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, AutoAdapterModel

"""Model components for the MetaGraphSci architecture.

The network keeps text, metadata, and citation context as separate streams until
late fusion. The main architectural update in this file is the citation branch:
it no longer relies on fixed hop buckets or PageRank-oriented ranking. Instead,
it uses a learned local selector and a compact citation transformer over a
bounded candidate set.

- shared SciBERT encoding for center and citation papers,
- lightweight metadata encoder for structured fields,
- learned citation selection before transformer aggregation,
- fusion-ready multimodal embeddings with clean ablation support.
"""

TensorTriplet: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
ABLATION_MODES = {
    "full": {"text", "metadata", "citation"}, 
    "text_only": {"text"}, 
    "text_metadata": {"text", "metadata"}, 
    "text_citation": {"text", "citation"}
}


class TextEncoder(nn.Module):
    """Adapts a pre-trained SciBERT model for feature extraction.
    
    Manages precision limits, Low-Rank Adaptation (LoRA), and partial backbone 
    freezing to strictly constrain memory usage while allowing the shared text 
    backbone to fine-tune its deeper layers.
    """

    def __init__(
        self,
        model_name: str, output_dim: int, peft_mode: str,
        lora_r: int, lora_alpha: int, lora_dropout: float,
        peft_target_modules: tuple[str, ...] | None,
        gradient_checkpointing: bool, freeze_backbone_until_layer: int,
        torch_dtype: torch.dtype | None = None,
        low_cpu_mem_usage: bool = True) -> None:
        super().__init__()
        peft_mode = peft_mode.lower()

        if model_name != "allenai/scibert":
            raise ValueError("This encoder is intended for SciBERT only.")

        # Default to bfloat16 to avoid NaN spikes during mixed-precision training if hardware supports it
        optype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        # Intercept weight loading to quantize the base model on-the-fly for QLoRA
        quantization_config = None
        if peft_mode == "qlora":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=optype)

        self.backbone_pretrained = AutoAdapterModel.from_pretrained(
            model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype, quantization_config=quantization_config)
        self.backbone = self.backbone_pretrained.load_adapter("allenai/scibert", source="hf", set_active=True)

        hidden_size = int(self.backbone.config.hidden_size)

        # Trade computation time for VRAM by dropping intermediate activations during the forward pass
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()

        if peft_mode == "qlora":
            self.backbone = prepare_model_for_kbit_training(
                self.backbone, use_gradient_checkpointing=gradient_checkpointing)

        # Lock the lower layers to retain generalized linguistic features
        if freeze_backbone_until_layer > 0:
            self.freeze_backbone(freeze_backbone_until_layer)

        # Inject trainable low-rank matrices into the attention mechanisms
        if peft_mode in {"lora", "qlora"}:
            target_modules = peft_target_modules or ("query", "value")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=list(target_modules), bias="none")
            self.backbone = get_peft_model(self.backbone, peft_config)

        self.projection = None
        if hidden_size != output_dim:
            self.projection = nn.Linear(hidden_size, output_dim)

    def freeze_backbone(self, num_layers: int) -> None:
        """Disables gradient tracking for the word embeddings and the bottom N transformer layers."""
        base_model = self.backbone
        if hasattr(self.backbone, "base_model_prefix"):
            base_model = getattr(self.backbone, self.backbone.base_model_prefix, self.backbone)

        embeddings = getattr(base_model, "embeddings", None)
        if embeddings is not None:
            for parameter in embeddings.parameters():
                parameter.requires_grad = False

        encoder = getattr(base_model, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is not None:
            for layer in list(layers)[:max(0, num_layers)]:
                for parameter in layer.parameters():
                    parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extracts the semantic representation of the document using the [CLS] token."""
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask).last_hidden_state

        cls_embedding = hidden_states[:, 0]
        if self.projection is not None:
            cls_embedding = self.projection(cls_embedding)
        return cls_embedding


class DeepCrossNetwork(nn.Module):
    """Models explicit, bounded-degree interactions between categorical metadata features.
    
    Instead of relying solely on deep MLPs to implicitly learn relationships, this layer 
    explicitly computes polynomial feature crosses at each step using the formula:
    $x_{l+1} = x_0 (x_l^T w_l) + b_l + x_l$
    """

    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim)) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for weight, bias in zip(self.weights, self.biases):
            # The inner product acts as an attention-like scalar that scales the original input,
            # retaining the original feature identities while discovering interactions.
            x = x0 * torch.sum(x * weight, dim=1, keepdim=True) + bias + x
        return x


class MetadataEncoder(nn.Module):
    """Maps discrete publication metadata into a dense continuous space.
    
    Aggregates variable-length author lists via masked mean pooling, concatenates them 
    with venue, publisher, and year, and feeds the joint vector into the Deep Cross Network.
    """

    def __init__(self, num_venues: int, num_publishers: int, num_authors: int, embedding_dim: int, cross_layers: int, output_dim: int) -> None:
        super().__init__()
        metadata_feature_dim = embedding_dim * 4
        self.venue = nn.Embedding(num_venues, embedding_dim)
        self.publisher = nn.Embedding(num_publishers, embedding_dim)
        self.author = nn.Embedding(num_authors, embedding_dim, padding_idx=0)
        self.year = nn.Linear(1, embedding_dim)
        self.cross = DeepCrossNetwork(metadata_feature_dim, cross_layers)
        self.projection = nn.Sequential(nn.Linear(metadata_feature_dim, output_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(output_dim, output_dim))

    def forward(self, venue_ids: torch.Tensor, publisher_ids: torch.Tensor, author_ids: torch.Tensor, years: torch.Tensor) -> torch.Tensor:
        years = years.float().view(-1, 1)
        author_embeddings = self.author(author_ids)
        
        # Masked average pooling prevents padded zeros in the author sequence from skewing the mean
        author_mask = (author_ids != 0).float().unsqueeze(-1)
        author_mean = (author_embeddings * author_mask).sum(dim=1) / author_mask.sum(dim=1).clamp_min(1.0)
        
        features = torch.cat([self.venue(venue_ids), self.publisher(publisher_ids), author_mean, self.year(years)], dim=1)
        return self.projection(self.cross(features))


class LearnedCitationSelector(nn.Module):
    """Acts as a dynamic attention bottleneck for the citation graph.
    
    Rather than processing all possible citations, it computes a relevance score 
    for each candidate using text semantics, graph topology (edge types), and publication gap.
    """

    def __init__(self, text_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_proj = nn.Linear(text_dim, hidden_dim)
        self.candidate_proj = nn.Linear(text_dim, hidden_dim)
        self.edge_embedding = nn.Embedding(4, hidden_dim)
        self.time_proj = nn.Linear(1, hidden_dim)
        self.score_proj = nn.Linear(1, hidden_dim)
        self.scorer = nn.Sequential(nn.Linear(hidden_dim * 5 + 1, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(
        self, center_text: torch.Tensor,
        candidate_embeddings: torch.Tensor, candidate_mask: torch.Tensor,
        edge_types: torch.Tensor, year_deltas: torch.Tensor, 
        cache_scores: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        _, num_candidates, _ = candidate_embeddings.shape
        
        # Broadcast the center paper's text representation across the candidate sequence dimension
        query = self.query_proj(center_text).unsqueeze(1).expand(-1, num_candidates, -1)
        candidate = self.candidate_proj(candidate_embeddings)
        edge = self.edge_embedding(edge_types.clamp_min(0))
        time = self.time_proj(year_deltas.unsqueeze(-1))
        prior = self.score_proj(cache_scores.unsqueeze(-1))
        
        # Provide the raw semantic cosine similarity as a direct scalar feature
        semantic = F.cosine_similarity(center_text.unsqueeze(1), candidate_embeddings, dim=-1).unsqueeze(-1)

        features = torch.cat([query, candidate, edge, time, prior, semantic], dim=-1)
        logits = self.scorer(features).squeeze(-1)
        
        # Forcibly reject padding candidates by driving their logits to negative infinity
        logits = logits.masked_fill(~candidate_mask.bool(), -1e9)

        k = max(1, min(int(top_k), num_candidates))
        top_logits, top_idx = logits.topk(k=k, dim=1)
        return top_idx, top_logits, logits


class StructuralCitationTokenizer(nn.Module):
    """Enriches raw textual embeddings of citations with graph-aware positional encodings.
    
    Transforms isolated text representations into tokens that inherently know *how* they 
    connect to the center document (e.g., direct citation vs. co-citation, publication gap, hop distance).
    """

    def __init__(self, text_dim: int, token_dim: int, hop_profile_dim: int = 2, spectral_dim: int = 0) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_dim, token_dim)
        self.edge_embedding = nn.Embedding(4, token_dim)
        self.time_proj = nn.Linear(1, token_dim)
        self.score_proj = nn.Linear(1, token_dim)
        self.hop_proj = nn.Linear(max(hop_profile_dim, 1), token_dim)
        self.spectral_proj = nn.Linear(max(spectral_dim, 1), token_dim)
        self.hop_profile_dim = int(hop_profile_dim)
        self.spectral_dim = int(spectral_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        candidate_embeddings: torch.Tensor,
        edge_types: torch.Tensor, year_deltas: torch.Tensor,
        selector_scores: torch.Tensor, hop_profiles: torch.Tensor | None = None,
        spectral_features: torch.Tensor | None = None) -> torch.Tensor:
        
        text_part = self.text_proj(candidate_embeddings)
        structural_part = self.edge_embedding(edge_types.clamp_min(0)) + self.time_proj(year_deltas.unsqueeze(-1)) + self.score_proj(selector_scores.unsqueeze(-1))
        
        # Generate neutral dummy tensors if complex structural profiles are disabled in the config
        if hop_profiles is None:
            hop_profiles = candidate_embeddings.new_zeros(candidate_embeddings.size(0), candidate_embeddings.size(1), self.hop_profile_dim)
        if spectral_features is None:
            spectral_features = candidate_embeddings.new_zeros(candidate_embeddings.size(0), candidate_embeddings.size(1), self.spectral_dim)

        hop_input = hop_profiles if self.hop_profile_dim > 0 else candidate_embeddings.new_zeros(candidate_embeddings.size(0), candidate_embeddings.size(1), 1)
        spectral_input = spectral_features if self.spectral_dim > 0 else candidate_embeddings.new_zeros(candidate_embeddings.size(0), candidate_embeddings.size(1), 1)
        
        structural_part = structural_part + self.hop_proj(hop_input) + self.spectral_proj(spectral_input)
        
        # Merge text and structure additively, applying layer normalization to stabilize variance
        return self.norm(text_part + structural_part)


class RelationAwareTransformerLayer(nn.Module):
    """A specialized multi-head attention block for reasoning over the citation graph.
    
    Unlike standard self-attention, this layer injects explicit relational matrices 
    (structural edge distances, metadata compatibility) directly into the pre-softmax 
    attention scores, allowing the network to modulate attention based on topological rules.
    """

    def __init__(self, token_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = token_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.out_proj = nn.Linear(token_dim, token_dim)
        self.structure_bias = nn.Linear(3, num_heads)
        self.metadata_bias = nn.Linear(1, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ff = nn.Sequential(nn.Linear(token_dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, token_dim), nn.Dropout(dropout))

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Splits the embedding dimension into multiple attention heads."""
        batch_size, seq_len, dim = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None, metadata_compatibility: torch.Tensor | None = None) -> torch.Tensor:
        
        residual = tokens
        normalized = self.norm1(tokens)

        q = self._reshape_heads(self.q_proj(normalized))
        k = self._reshape_heads(self.k_proj(normalized))
        v = self._reshape_heads(self.v_proj(normalized))

        # Core query-key similarity matrix
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Integrate relational and metadata heuristics into the attention distribution
        if attention_bias is not None:
            logits = logits + self.structure_bias(attention_bias).permute(0, 3, 1, 2)
        if metadata_compatibility is not None:
            logits = logits + self.metadata_bias(metadata_compatibility.unsqueeze(-1)).permute(0, 3, 1, 2)
            
        # Eliminate connections to padded graph nodes
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], -1e9)

        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        
        # Apply attention weights to values and project back to original dimensions
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous().view(tokens.size(0), tokens.size(1), tokens.size(2))
        tokens = residual + self.dropout(self.out_proj(attended))
        return tokens + self.ff(self.norm2(tokens))


class CitationGraphTransformer(nn.Module):
    """Orchestrates the selection and contextualization of the local citation neighborhood.
    
    Execution pipeline:
    1. Pass raw citation text through the shared SciBERT encoder.
    2. Score candidates against the center query and keep the top-k highest quality nodes.
    3. Construct a query matrix integrating relational heuristics.
    4. Run transformer layers over the resulting structure-aware citation tokens.
    """

    def __init__(
        self,
        text_dim: int, output_dim: int, metadata_dim: int,
        num_heads: int, num_layers: int,
        ff_dim: int, selector_hidden_dim: int, selector_top_k: int,
        max_context_size: int, dropout: float, hop_profile_dim: int = 2, spectral_dim: int = 0) -> None:
        super().__init__()
        self.selector_top_k = int(min(selector_top_k, max_context_size))
        self.center_proj = nn.Linear(text_dim, output_dim)
        self.selector = LearnedCitationSelector(text_dim, selector_hidden_dim, dropout=dropout)
        self.tokenizer = StructuralCitationTokenizer(text_dim, output_dim, hop_profile_dim=hop_profile_dim, spectral_dim=spectral_dim)
        self.layers = nn.ModuleList([RelationAwareTransformerLayer(output_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.meta_to_bias = nn.Linear(metadata_dim, output_dim)
        self.hop_profile_dim = int(hop_profile_dim)
        self.spectral_dim = int(spectral_dim)
        self.pool_gate = nn.Linear(output_dim * 2, output_dim)
        self.output_proj = nn.Sequential(nn.LayerNorm(output_dim), nn.Linear(output_dim, output_dim))

    @staticmethod
    def encode_candidates(candidate_input_ids: torch.Tensor, candidate_attention_mask: torch.Tensor, text_encoder: nn.Module) -> torch.Tensor:
        """Flattens the batch and sequence dimensions to push all candidates through the text backbone efficiently."""
        batch_size, max_candidates, seq_len = candidate_input_ids.shape
        flat_ids = candidate_input_ids.reshape(batch_size * max_candidates, seq_len)
        flat_mask = candidate_attention_mask.reshape(batch_size * max_candidates, seq_len)
        return text_encoder(flat_ids, flat_mask).reshape(batch_size, max_candidates, -1)

    @staticmethod
    def gather_last_dim(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Utility to pull dense vectors aligned with the indices produced by top-k selection."""
        expand = indices.unsqueeze(-1).expand(-1, -1, values.size(-1))
        return values.gather(1, expand)

    @staticmethod
    def gather_vector(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Utility to pull discrete scalar IDs aligned with top-k selection."""
        return values.gather(1, indices)

    def forward(
        self, center_text: torch.Tensor, center_metadata: torch.Tensor,
        candidate_embeddings: torch.Tensor, candidate_mask: torch.Tensor,
        edge_types: torch.Tensor, year_deltas: torch.Tensor, cache_scores: torch.Tensor,
        hop_profiles: torch.Tensor | None = None, spectral_features: torch.Tensor | None = None,
        context_metadata: torch.Tensor | None = None, context_years: torch.Tensor | None = None) -> torch.Tensor:
        
        top_idx, top_logits, _ = self.selector(
            center_text, candidate_embeddings, candidate_mask,
            edge_types, year_deltas, cache_scores, top_k=self.selector_top_k)

        # Slice out only the strongest topological candidates
        selected_embeddings = self.gather_last_dim(candidate_embeddings, top_idx)
        selected_mask = self.gather_vector(candidate_mask, top_idx)
        selected_edges = self.gather_vector(edge_types, top_idx)
        selected_years = self.gather_vector(year_deltas, top_idx)
        selected_scores = torch.sigmoid(top_logits)
        selected_hops = self.gather_last_dim(hop_profiles, top_idx) if hop_profiles is not None and hop_profiles.size(-1) > 0 else None
        selected_spectral = self.gather_last_dim(spectral_features, top_idx) if spectral_features is not None and spectral_features.size(-1) > 0 else None
        selected_context_meta = self.gather_last_dim(context_metadata, top_idx) if context_metadata is not None else None
        selected_context_years = self.gather_vector(context_years, top_idx) if context_years is not None else None

        # Assemble the input sequence: center paper token + context neighborhood tokens
        citation_tokens = self.tokenizer(selected_embeddings, selected_edges, selected_years, selected_scores, selected_hops, selected_spectral)
        citation_tokens = citation_tokens * selected_mask.unsqueeze(-1).float()

        center_token = self.center_proj(center_text).unsqueeze(1)
        tokens = torch.cat([center_token, citation_tokens], dim=1)
        
        # The first token (center) is never padded. Remaining tokens mask out missing neighbors.
        key_padding_mask = torch.cat([torch.zeros(center_text.size(0), 1, dtype=torch.bool, device=center_text.device), ~selected_mask.bool()], dim=1)

        batch_size, num_selected = selected_edges.shape
        total_tokens = num_selected + 1
        
        # Construct an explicit N x N attention bias matrix defining structural relationships
        attention_bias = tokens.new_zeros(batch_size, total_tokens, total_tokens, 3)
        if num_selected > 0:
            edge_bias = selected_edges.float() / 3.0
            year_bias = -selected_years.abs()
            if selected_hops is not None and selected_hops.size(-1) > 0:
                hop_bias = selected_hops.mean(dim=-1)
            else:
                hop_bias = selected_years.new_zeros(selected_years.shape)
            
            # Bias row 0 (center querying neighbors) and column 0 (neighbors querying center)
            attention_bias[:, 0, 1:, 0] = edge_bias
            attention_bias[:, 1:, 0, 0] = edge_bias
            attention_bias[:, 0, 1:, 1] = year_bias
            attention_bias[:, 1:, 0, 1] = year_bias
            attention_bias[:, 0, 1:, 2] = hop_bias
            attention_bias[:, 1:, 0, 2] = hop_bias

        # Inject publication compatibility limits into the attention bias
        metadata_compatibility = tokens.new_zeros(batch_size, total_tokens, total_tokens)
        if selected_context_meta is not None:
            center_meta_bias = self.meta_to_bias(center_metadata)
            compat = torch.sum(center_meta_bias.unsqueeze(1) * selected_context_meta, dim=-1) / max(selected_context_meta.size(-1), 1)
            metadata_compatibility[:, 0, 1:] = compat
            metadata_compatibility[:, 1:, 0] = compat
        if selected_context_years is not None:
            year_compat = 1.0 - torch.clamp((selected_context_years - selected_context_years.new_tensor(0.0)).abs(), min=0.0, max=1.0)
            metadata_compatibility[:, 0, 1:] = metadata_compatibility[:, 0, 1:] + year_compat
            metadata_compatibility[:, 1:, 0] = metadata_compatibility[:, 1:, 0] + year_compat

        # Process the aggregated graph structure
        for layer in self.layers:
            tokens = layer(tokens, key_padding_mask=key_padding_mask, attention_bias=attention_bias, metadata_compatibility=metadata_compatibility)

        # Distill output via a learned gating mechanism blending the center token with pooled neighbor representations
        center_out = tokens[:, 0]
        pooled = (tokens[:, 1:] * selected_mask.unsqueeze(-1).float()).sum(dim=1) / selected_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        gate = torch.sigmoid(self.pool_gate(torch.cat([center_out, pooled], dim=-1)))
        mixed = center_out + gate * pooled
        
        return self.output_proj(mixed)


class MultimodalFusion(nn.Module):
    """Combines isolated modalities into a single classification-ready vector.
    
    Uses a gated residual block to non-linearly blend features, supporting heavy 
    modality dropout during training to prevent over-reliance on any single stream.
    """

    def __init__(self, text_dim: int, metadata_dim: int, citation_dim: int, fusion_dim: int, modality_dropout: float) -> None:
        super().__init__()
        total_dim = text_dim + metadata_dim + citation_dim
        self.modality_dropout = float(modality_dropout)
        self.input_proj = nn.Sequential(nn.Linear(total_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        self.fuser = nn.Sequential(nn.Linear(total_dim, fusion_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(fusion_dim, fusion_dim))
        self.gate = nn.Linear(total_dim, fusion_dim)

    def maybe_drop(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zeros out an entire modality stream during training and rescales."""
        if self.training and self.modality_dropout > 0.0:
            keep = torch.bernoulli(torch.full((x.size(0), 1), 1.0 - self.modality_dropout, device=x.device))
            x = x * keep / max(1.0 - self.modality_dropout, 1e-6)
        return x

    def forward(self, h_text: torch.Tensor, h_meta: torch.Tensor, h_citation: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([h_text, self.maybe_drop(h_meta), self.maybe_drop(h_citation)], dim=1)
        residual = self.input_proj(concatenated)
        mixed = self.fuser(concatenated)
        gate = torch.sigmoid(self.gate(concatenated))
        return residual + gate * mixed


class NormalizedCosineClassifier(nn.Module):
    """Classifies inputs by measuring cosine similarity to learnable prototypes.
    
    Projects both the fused embedding and the target class weights onto a unit hypersphere, 
    computing $cos(\theta)$, which creates well-separated clusters resistant to magnitude scaling.
    """

    def __init__(self, input_dim: int, num_classes: int, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.class_vectors = nn.Parameter(torch.empty(num_classes, input_dim))
        nn.init.normal_(self.class_vectors, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_x = F.normalize(x, p=2, dim=1)
        normalized_prototypes = F.normalize(self.class_vectors, p=2, dim=1)
        
        # Scaling parameter adjusts the sharpness of the logits distribution
        logits = self.scale * (normalized_x @ normalized_prototypes.t())
        return logits, F.softmax(logits, dim=1)


class MetaGraphSci(nn.Module):
    """Citation-aware multimodal encoder for scientific document classification.
    
    Orchestrates the entire forward pass by driving inputs through the Text, 
    Metadata, and Citation encoders, then fusing them for final classification.
    """

    def __init__(
        self,
        num_classes: int, num_venues: int, num_publishers: int, num_authors: int,
        text_dim: int, metadata_dim: int, citation_dim: int, fusion_dim: int,
        classifier_scale: float, model_name: str, ablation_mode: str, peft_mode: str,
        lora_r: int, lora_alpha: int, lora_dropout: float, peft_target_modules: tuple[str, ...] | None,
        gradient_checkpointing: bool, freeze_backbone_until_layer: int,
        citation_heads: int, citation_layers: int, citation_ff_dim: int,
        selector_hidden_dim: int, selector_top_k: int,
        max_context_size: int, fusion_modality_dropout: float,
        metadata_embedding_dim: int = 64, metadata_cross_layers: int = 3, citation_dropout: float = 0.1,
        hop_profile_dim: int = 2, spectral_dim: int = 0) -> None:
        
        super().__init__()
        self.default_ablation_mode = ablation_mode
        
        self.text_encoder = TextEncoder(
            model_name, text_dim, peft_mode, lora_r, lora_alpha, lora_dropout,
            peft_target_modules, gradient_checkpointing, freeze_backbone_until_layer)
            
        self.metadata_encoder = MetadataEncoder(
            num_venues, num_publishers, num_authors, metadata_embedding_dim, metadata_cross_layers, metadata_dim)
            
        self.citation_encoder = CitationGraphTransformer(
            text_dim, citation_dim, metadata_dim, citation_heads, citation_layers, citation_ff_dim,
            selector_hidden_dim, selector_top_k, max_context_size, citation_dropout,
            hop_profile_dim=hop_profile_dim, spectral_dim=spectral_dim)
            
        self.fusion = MultimodalFusion(text_dim, metadata_dim, citation_dim, fusion_dim, fusion_modality_dropout)
        self.classifier = NormalizedCosineClassifier(fusion_dim, num_classes, classifier_scale)

    def ablation_study(self, h_text: torch.Tensor, h_meta: torch.Tensor, h_citation: torch.Tensor, mode: str) -> TensorTriplet:
        """Forces the dimensions of inactive modalities to zero during ablation studies."""
        keep_modalities = ABLATION_MODES.get(mode, ABLATION_MODES["full"])
        tensors, names = (h_text, h_meta, h_citation), ("text", "metadata", "citation")
        return tuple(tensor if name in keep_modalities else tensor.new_zeros(tensor.shape) for name, tensor in zip(names, tensors))

    def encode_modalities(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        venue_ids: torch.Tensor, publisher_ids: torch.Tensor, author_ids: torch.Tensor, years: torch.Tensor,
        context_input_ids: torch.Tensor, context_attention_mask: torch.Tensor, context_mask: torch.Tensor,
        context_edge_types: torch.Tensor,  context_year_deltas: torch.Tensor, context_scores: torch.Tensor,
        context_hop_profiles: torch.Tensor | None = None, context_spectral: torch.Tensor | None = None,
        context_venue_ids: torch.Tensor | None = None, context_publisher_ids: torch.Tensor | None = None, context_years: torch.Tensor | None = None,
        ablation_mode: str | None = None) -> TensorTriplet:
        
        h_text = self.text_encoder(input_ids, attention_mask)
        h_meta = self.metadata_encoder(venue_ids, publisher_ids, author_ids, years)
        candidate_embeddings = self.citation_encoder.encode_candidates(context_input_ids, context_attention_mask, self.text_encoder)
        
        # We need metadata representations for candidate citations to construct the compatibility bias. 
        # Fake authors array creates neutral padded inputs since neighbor authors aren't retrieved.
        context_meta = None
        if context_venue_ids is not None and context_publisher_ids is not None and context_years is not None:
            fake_authors = context_venue_ids.new_zeros(context_venue_ids.size(0), context_venue_ids.size(1), author_ids.size(1))
            flat_meta = self.metadata_encoder(
                context_venue_ids.reshape(-1), context_publisher_ids.reshape(-1), 
                fake_authors.reshape(-1, fake_authors.size(-1)), context_years.reshape(-1, 1)
            )
            context_meta = flat_meta.reshape(context_venue_ids.size(0), context_venue_ids.size(1), -1)
            
        h_citation = self.citation_encoder(
            h_text, h_meta, candidate_embeddings, context_mask, context_edge_types, context_year_deltas, 
            context_scores, context_hop_profiles, context_spectral, context_meta, context_years)
            
        return self.ablation_study(h_text, h_meta, h_citation, ablation_mode or self.default_ablation_mode)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        venue_ids: torch.Tensor, publisher_ids: torch.Tensor, author_ids: torch.Tensor, years: torch.Tensor,
        context_input_ids: torch.Tensor, context_attention_mask: torch.Tensor, context_mask: torch.Tensor,
        context_edge_types: torch.Tensor, context_year_deltas: torch.Tensor, context_scores: torch.Tensor,
        context_hop_profiles: torch.Tensor | None = None, context_spectral: torch.Tensor | None = None,
        context_venue_ids: torch.Tensor | None = None, context_publisher_ids: torch.Tensor | None = None, context_years: torch.Tensor | None = None,
        ablation_mode: str | None = None, return_parts: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        
        h_text, h_meta, h_citation = self.encode_modalities(
            input_ids, attention_mask, venue_ids, publisher_ids, author_ids, years,
            context_input_ids, context_attention_mask, context_mask, context_edge_types, context_year_deltas, context_scores,
            context_hop_profiles=context_hop_profiles, context_spectral=context_spectral,
            context_venue_ids=context_venue_ids, context_publisher_ids=context_publisher_ids, context_years=context_years,
            ablation_mode=ablation_mode)
            
        embeddings = self.fusion(h_text, h_meta, h_citation)
        logits, probabilities = self.classifier(embeddings)
        
        if return_parts:
            return embeddings, logits, probabilities, {"text": h_text, "metadata": h_meta, "citation": h_citation}
        return embeddings, logits, probabilities

    def get_embeddings(self, ablation_mode: str | None = None, **batch: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper used by the contrastive learning trainer to fetch raw representations."""
        return self.forward(ablation_mode=ablation_mode, **batch)[0]
