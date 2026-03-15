import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ablation import NUM_RELATIONS


class LearnedCitationSelector(nn.Module):
    """Dynamic attention bottleneck for filtering the citation graph."""

    def __init__(self, text_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        #  Dynamic Attention Bottleneck
        # The `max_context_size` cache limits memory during data loading, but passing 
        # e.g., 64 citations through deep Transformer layers is still computationally massive.
        # This module acts as a fast, learned bottleneck. It projects all candidates into a 
        # lightweight semantic space, scores them against the anchor text, and hard-prunes 
        # the context down to the Top-K most relevant neighbors *before* they enter the heavy 
        # Relation-Aware Transformer blocks.
        self.query_proj = nn.Linear(text_dim, hidden_dim)
        self.candidate_proj = nn.Linear(text_dim, hidden_dim)
        self.edge_embedding = nn.Embedding(4, hidden_dim)
        self.time_proj = nn.Linear(1, hidden_dim)
        self.score_proj = nn.Linear(1, hidden_dim)
        
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 1, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(
        self, center_text: Tensor, candidate_embeddings: Tensor, candidate_mask: Tensor,
        edge_types: Tensor, year_deltas: Tensor, cache_scores: Tensor, top_k: int) -> tuple[Tensor, Tensor, Tensor]:
        _, num_candidates, _ = candidate_embeddings.shape
        
        query = self.query_proj(center_text).unsqueeze(1).expand(-1, num_candidates, -1)
        candidate = self.candidate_proj(candidate_embeddings)
        edge = self.edge_embedding(edge_types.clamp_min(0))
        time = self.time_proj(year_deltas.unsqueeze(-1))
        prior = self.score_proj(cache_scores.unsqueeze(-1))
        semantic = F.cosine_similarity(center_text.unsqueeze(1), candidate_embeddings, dim=-1).unsqueeze(-1)
        
        features = torch.cat([query, candidate, edge, time, prior, semantic], dim=-1)
        logits = self.scorer(features).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask.bool(), -1e9)
        
        k = max(1, min(int(top_k), num_candidates))
        top_logits, top_idx = logits.topk(k=k, dim=1)
        return top_idx, top_logits, logits


class StructuralCitationTokenizer(nn.Module):
    """Enriches raw citation embeddings with graph-aware positional encodings."""

    def __init__(self, text_dim: int, token_dim: int, hop_profile_dim: int = 2, spectral_dim: int = 0) -> None:
        super().__init__()
        # Structural Positional Encodings
        # Unlike text sequences where "position" is just an integer (1, 2, 3...), 
        # a citation graph node has multi-dimensional positional coordinates:
        # 1. Edge Type (Incoming vs. Outgoing)
        # 2. Temporal Distance (Years apart)
        # 3. Global Coordinates (Spectral Eigenvectors)
        # reshape_heads linearly project all these structural traits and sum them into the text embedding, 
        # transforming a standard BERT vector into a true "Graph Token".
        self.hop_profile_dim = int(hop_profile_dim)
        self.spectral_dim = int(spectral_dim)
        
        self.text_proj = nn.Linear(text_dim, token_dim)
        self.edge_embedding = nn.Embedding(4, token_dim)
        self.time_proj = nn.Linear(1, token_dim)
        self.score_proj = nn.Linear(1, token_dim)
        self.hop_proj = nn.Linear(max(hop_profile_dim, 1), token_dim)
        self.spectral_proj = nn.Linear(max(spectral_dim, 1), token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self, candidate_embeddings: Tensor, edge_types: Tensor, year_deltas: Tensor,
        selector_scores: Tensor, hop_profiles: Tensor | None = None, spectral_features: Tensor | None = None) -> Tensor:
        text_part = self.text_proj(candidate_embeddings)
        structural_part = (
            self.edge_embedding(edge_types.clamp_min(0)) + 
            self.time_proj(year_deltas.unsqueeze(-1)) + 
            self.score_proj(selector_scores.unsqueeze(-1)))

        b_size, seq_len, _ = candidate_embeddings.shape
        if hop_profiles is None: 
            hop_profiles = candidate_embeddings.new_zeros(b_size, seq_len, self.hop_profile_dim)
        if spectral_features is None: 
            spectral_features = candidate_embeddings.new_zeros(b_size, seq_len, self.spectral_dim)

        hop_input = hop_profiles if self.hop_profile_dim > 0 else candidate_embeddings.new_zeros(b_size, seq_len, 1)
        spectral_input = spectral_features if self.spectral_dim > 0 else candidate_embeddings.new_zeros(b_size, seq_len, 1)

        structural_part = structural_part + self.hop_proj(hop_input) + self.spectral_proj(spectral_input)
        return self.norm(text_part + structural_part)


class LatentGraphModule(nn.Module):
    """Learns a sparse latent adjacency matrix among selected context tokens."""

    def __init__(self, token_dim: int, top_k: int = 4) -> None:
        super().__init__()
        # Latent Graph Discovery (Virtual Edges)
        # Citation graphs are notoriously incomplete (missing edges) and noisy (false edges). 
        # By computing pairwise similarity (Query-Key dot products) between all neighbor 
        # tokens, reshape_headsallow the model to discover "Virtual Edges"-connections between two 
        # papers in the local context that don't cite each other, but discuss identical topics. 
        # Top-K sparsification forces this latent graph to remain focused rather than fully connected.
        self.q_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.k_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.top_k = int(top_k)
        self.scale = token_dim ** -0.5

    def forward(self, neighbor_tokens: Tensor, neighbor_mask: Tensor) -> Tensor:
        B, N, _ = neighbor_tokens.shape
        q = self.q_proj(neighbor_tokens)
        k = self.k_proj(neighbor_tokens)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

        invalid = ~neighbor_mask.bool()
        scores = scores.masked_fill(invalid.unsqueeze(1), -1e9)
        scores = scores.masked_fill(invalid.unsqueeze(2), -1e9)
        scores = scores.masked_fill(torch.eye(N, dtype=torch.bool, device=scores.device).unsqueeze(0), -1e9)

        effective_k = min(self.top_k, N - 1)
        if effective_k > 0 and N > 1:
            topk_vals, _ = scores.topk(effective_k, dim=-1)
            threshold = topk_vals[..., -1:].detach()
            scores = scores.masked_fill(scores < threshold, -1e9)

        latent_adj = torch.softmax(scores, dim=-1)
        return latent_adj * neighbor_mask.float().unsqueeze(-1)


class RelationMixer(nn.Module):
    """Dynamically mixes diverse relation types into a unified attention bias."""

    def __init__(self, num_relations: int, num_heads: int) -> None:
        super().__init__() 
        # Adaptive Relation Attention
        # The model has multiple overlapping graphs (Citation, Temporal, Metadata, Latent). 
        # reshape_headsdefine a learnable `mixing_logits` vector. The Softmax over these logits dictates 
        # how much "weight" each graph topology contributes to the final attention bias matrix. 
        # This allows different transformer heads to independently specialize—e.g., Head 0 
        # might strictly follow Citation edges, while Head 1 strictly follows the Temporal graph.
        self.num_relations = int(num_relations)
        self.mixing_logits = nn.Parameter(torch.zeros(num_relations))
        self.head_proj = nn.Linear(num_relations, num_heads, bias=False)

    def forward(self, relation_indicators: Tensor) -> Tensor:
        alpha = torch.softmax(self.mixing_logits, dim=0)
        weighted = relation_indicators * alpha
        return self.head_proj(weighted).permute(0, 3, 1, 2)


class RelationAwareTransformerLayer(nn.Module):
    """Multi-head attention layer specialized for relational graph reasoning."""

    def __init__(self, token_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        if token_dim % num_heads != 0: 
            raise ValueError("token_dim must be divisible by num_heads")
            
        self.num_heads = int(num_heads)
        self.head_dim = token_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm1 = nn.LayerNorm(token_dim)
        self.q_proj, self.k_proj, self.v_proj = (
            nn.Linear(token_dim, token_dim), 
            nn.Linear(token_dim, token_dim),
            nn.Linear(token_dim, token_dim))
        self.out_proj = nn.Linear(token_dim, token_dim)
        
        self.structure_bias = nn.Linear(3, num_heads)
        self.metadata_bias = nn.Linear(1, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(token_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(token_dim, ff_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, token_dim), 
            nn.Dropout(dropout))

    def reshape_heads(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, tokens: Tensor, key_padding_mask: Tensor | None = None,
        attention_bias: Tensor | None = None, metadata_compatibility: Tensor | None = None,
        latent_adj_bias: Tensor | None = None, multi_relation_bias: Tensor | None = None) -> Tensor:

        residual, normalized = tokens, self.norm1(tokens)
        q, k, v = (
            self.reshape_heads(self.q_proj(normalized)), 
            self.reshape_heads(self.k_proj(normalized)), 
            self.reshape_heads(self.v_proj(normalized)))
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Additive Attention Bias
        # Standard transformers compute attention purely from node features (Q * K). 
        # By directly *adding* these structural bias matrices to the attention logits 
        # before the Softmax, reshape_heads mathematically guarantee that graph topology (edges, years) 
        # explicitly governs the flow of information between nodes, creating a Graph Transformer.
        if attention_bias is not None: 
            logits = logits + self.structure_bias(attention_bias).permute(0, 3, 1, 2)
        if metadata_compatibility is not None: 
            logits = logits + self.metadata_bias(metadata_compatibility.unsqueeze(-1)).permute(0, 3, 1, 2)
        if latent_adj_bias is not None: 
            logits = logits + latent_adj_bias
        if multi_relation_bias is not None: 
            logits = logits + multi_relation_bias
        if key_padding_mask is not None: 
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], -1e9)

        weights = self.dropout(torch.softmax(logits, dim=-1))
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous().view(tokens.size(0), tokens.size(1), tokens.size(2))
        
        tokens = residual + self.dropout(self.out_proj(attended))
        return tokens + self.ff(self.norm2(tokens))


class CitationGraphTransformer(nn.Module):
    """Orchestrates selection and multi-relational contextualization of the citation neighborhood."""

    def __init__(
        self, text_dim: int, output_dim: int, metadata_dim: int, num_heads: int, num_layers: int, ff_dim: int,
        selector_hidden_dim: int, selector_top_k: int, max_context_size: int, dropout: float,
        hop_profile_dim: int = 2, spectral_dim: int = 0, use_latent_graph: bool = True, latent_graph_top_k: int = 4,
        num_relations: int = NUM_RELATIONS, hybrid_alpha_init: float = 0.0) -> None:
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

        self.use_latent_graph = bool(use_latent_graph)
        if use_latent_graph:
            self.latent_graph = LatentGraphModule(output_dim, top_k=latent_graph_top_k)
            self.latent_adj_head_proj = nn.Linear(1, num_heads, bias=False)
            self.hybrid_alpha = nn.Parameter(Tensor([float(hybrid_alpha_init)]))
        else:
            self.latent_graph, self.latent_adj_head_proj, self.hybrid_alpha = None, None, None

        self.num_relations = int(num_relations)
        self.relation_mixer = RelationMixer(num_relations, num_heads)
        self._num_heads = int(num_heads)

    @staticmethod
    def encode_candidates(candidate_input_ids: Tensor, candidate_attention_mask: Tensor, text_encoder: nn.Module) -> Tensor:
        B, M, S = candidate_input_ids.shape
        flat_ids, flat_mask = candidate_input_ids.reshape(B * M, S), candidate_attention_mask.reshape(B * M, S)
        return text_encoder(flat_ids, flat_mask).reshape(B, M, -1)

    @staticmethod
    def gather_last_dim(values: Tensor, indices: Tensor) -> Tensor:
        return values.gather(1, indices.unsqueeze(-1).expand(-1, -1, values.size(-1)))

    @staticmethod
    def gather_vector(values: Tensor, indices: Tensor) -> Tensor:
        return values.gather(1, indices)

    def build_relation_indicators(
        self, selected_edges: Tensor, selected_years: Tensor, center_metadata: Tensor,
        context_meta: Tensor | None, latent_adj: Tensor | None, tokens: Tensor) -> Tensor:
        # Prepares the 4 specific relation types for the RelationMixer
        B, N = selected_edges.shape
        total = N + 1
        indicators = tokens.new_zeros(B, total, total, self.num_relations)

        edge_norm = selected_edges.float() / 3.0
        indicators[:, 0, 1:, 0] = edge_norm
        indicators[:, 1:, 0, 0] = edge_norm

        year_prox = torch.exp(-selected_years.abs().float() / 5.0)
        indicators[:, 0, 1:, 1] = year_prox
        indicators[:, 1:, 0, 1] = year_prox

        if context_meta is not None and self.num_relations >= 3:
            c_norm, n_norm = F.normalize(center_metadata.float(), dim=-1), F.normalize(context_meta.float(), dim=-1)
            meta_compat = (c_norm.unsqueeze(1) * n_norm).sum(-1).clamp(-1.0, 1.0)
            indicators[:, 0, 1:, 2], indicators[:, 1:, 0, 2] = meta_compat, meta_compat

        if latent_adj is not None and self.num_relations >= 4:
            indicators[:, 1:, 1:, 3] = latent_adj

        return indicators

    def forward(
        self, center_text: Tensor, center_metadata: Tensor, candidate_embeddings: Tensor,
        candidate_mask: Tensor, edge_types: Tensor, year_deltas: Tensor, cache_scores: Tensor,
        hop_profiles: Tensor | None = None, spectral_features: Tensor | None = None,
        context_metadata: Tensor | None = None, context_years: Tensor | None = None) -> Tensor:

        # Prune the context neighborhood down to Top-K via Learned Bottleneck
        top_idx, top_logits, _ = self.selector(
            center_text, candidate_embeddings, candidate_mask,
            edge_types, year_deltas, cache_scores, top_k=self.selector_top_k)

        selected_embeddings = self.gather_last_dim(candidate_embeddings, top_idx)
        selected_mask, selected_edges, selected_years = (
            self.gather_vector(candidate_mask, top_idx),
            self.gather_vector(edge_types, top_idx), 
            self.gather_vector(year_deltas, top_idx))
        selected_scores = torch.sigmoid(top_logits)
        
        selected_hops = self.gather_last_dim(hop_profiles, top_idx) \
            if hop_profiles is not None and hop_profiles.size(-1) > 0 else None
        selected_spectral = self.gather_last_dim(spectral_features, top_idx) \
            if spectral_features is not None and spectral_features.size(-1) > 0 else None
        selected_context_meta = self.gather_last_dim(context_metadata, top_idx) \
            if context_metadata is not None else None
        selected_context_years = self.gather_vector(context_years, top_idx) \
            if context_years is not None else None

        # Tokenize surviving nodes into Graph Tokens and construct the sequence
        citation_tokens = self.tokenizer(
            selected_embeddings, selected_edges,
            selected_years, selected_scores, selected_hops, selected_spectral
        ) * selected_mask.unsqueeze(-1).float()
        center_token = self.center_proj(center_text).unsqueeze(1)
        tokens = torch.cat([center_token, citation_tokens], dim=1)
        
        key_padding_mask = torch.cat([
            torch.zeros(center_text.size(0), 1, dtype=torch.bool, device=center_text.device),
            ~selected_mask.bool()], dim=1)
        batch_size, num_selected = selected_edges.shape
        total_tokens = num_selected + 1

        # Construct explicitly defined (hardcoded) structural biases
        attention_bias = tokens.new_zeros(batch_size, total_tokens, total_tokens, 3)
        if num_selected > 0:
            edge_bias, year_bias = selected_edges.float() / 3.0, -selected_years.abs()
            hop_bias = selected_hops.mean(dim=-1) \
                if selected_hops is not None and selected_hops.size(-1) > 0 else selected_years.new_zeros(selected_years.shape)
            
            attention_bias[:, 0, 1:, 0], attention_bias[:, 1:, 0, 0] = edge_bias, edge_bias
            attention_bias[:, 0, 1:, 1], attention_bias[:, 1:, 0, 1] = year_bias, year_bias
            attention_bias[:, 0, 1:, 2], attention_bias[:, 1:, 0, 2] = hop_bias, hop_bias

        metadata_compatibility = tokens.new_zeros(batch_size, total_tokens, total_tokens)
        if selected_context_meta is not None:
            compat = (self.meta_to_bias(center_metadata).unsqueeze(1) * selected_context_meta).sum(-1) / max(selected_context_meta.size(-1), 1)
            metadata_compatibility[:, 0, 1:], metadata_compatibility[:, 1:, 0] = compat, compat
            
        if selected_context_years is not None:
            year_compat = 1.0 - torch.clamp((selected_context_years - selected_context_years.new_tensor(0.0)).abs(), min=0.0, max=1.0)
            metadata_compatibility[:, 0, 1:] += year_compat
            metadata_compatibility[:, 1:, 0] += year_compat

        # Construct Latent Adjacency graph and route via Relation Mixer
        latent_adj = tokens.new_zeros(batch_size, num_selected, num_selected)
        latent_adj_full_bias = tokens.new_zeros(batch_size, self._num_heads, total_tokens, total_tokens)
        if self.use_latent_graph and self.latent_graph is not None and num_selected > 0:
            latent_adj = self.latent_graph(tokens[:, 1:], selected_mask.bool())
            alpha = torch.sigmoid(self.hybrid_alpha)
            latent_per_head = self.latent_adj_head_proj(latent_adj.unsqueeze(-1)).permute(0, 3, 1, 2)
            latent_adj_full_bias = tokens.new_zeros(batch_size, self._num_heads, total_tokens, total_tokens)
            latent_adj_full_bias[:, :, 1:, 1:] = (1.0 - alpha) * latent_per_head

        relation_indicators = self.build_relation_indicators(
            selected_edges, selected_years,
            center_metadata, selected_context_meta, latent_adj, tokens)
        multi_relation_bias = self.relation_mixer(relation_indicators)

        # Graph Message Passing
        for layer in self.layers:
            tokens = layer(
                tokens, key_padding_mask=key_padding_mask,
                attention_bias=attention_bias, metadata_compatibility=metadata_compatibility, 
                latent_adj_bias=latent_adj_full_bias, multi_relation_bias=multi_relation_bias)

        # Gated Output Pooling
        center_out = tokens[:, 0]
        pooled = (tokens[:, 1:] * selected_mask.unsqueeze(-1).float()).sum(dim=1) / selected_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        gate = torch.sigmoid(self.pool_gate(torch.cat([center_out, pooled], dim=-1)))
        return self.output_proj(center_out + gate * pooled)
