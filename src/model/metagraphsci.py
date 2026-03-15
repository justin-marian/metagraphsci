import torch.nn as nn
from torch import Tensor

from ablation import ABLATION_MODES, NUM_RELATIONS, TensorTriplet
from fusion_heads import MultimodalFusion, NormalizedCosineClassifier
from graph_encoder import CitationGraphTransformer
from metadata_encoder import MetadataEncoder
from text_encoder import TextEncoder


"""MetaGraphSci Architecture.

Multimodal graph neural network for scientific document classification. 
It combines textual representations, explicit metadata constraints, and scholarly graph topology 
to contextualize papers. 

Key Architectural Concepts:
- Hybrid Graph Construction: The model does not rely solely on the observed citation graph. 
    A LatentGraphModule learns pairwise affinities among selected context nodes, blending the 
    observed citation adjacency with learned connections. This allows the model to discover 
    same-topic but un-cited papers and cross-subfield bridges.
- Multi-Relation Mixing: The scholarly graph represents multiple types of relationships. 
    A RelationMixer combines these distinct relation types (e.g., citation edge, year-proximity, 
    metadata-compatibility, and learned latent similarity) into a unified attention bias, with 
    per-layer mixing weights that allow each transformer layer to focus on different relational aspects.
"""


class MetaGraphSci(nn.Module):
    """Citation-aware multimodal encoder for scientific document classification."""

    def __init__(
        self, num_classes: int, num_venues: int, num_publishers: int, num_authors: int,
        text_dim: int, metadata_dim: int, citation_dim: int, fusion_dim: int, classifier_scale: float,
        model_name: str, ablation_mode: str, peft_mode: str, lora_r: int, lora_alpha: int,
        lora_dropout: float, peft_target_modules: tuple[str, ...] | None, 
        gradient_checkpointing: bool, freeze_backbone_until_layer: int, 
        citation_heads: int, citation_layers: int, citation_ff_dim: int,
        selector_hidden_dim: int, selector_top_k: int, max_context_size: int, fusion_modality_dropout: float,
        metadata_embedding_dim: int = 64, metadata_cross_layers: int = 3, citation_dropout: float = 0.1,
        hop_profile_dim: int = 2, spectral_dim: int = 0, use_latent_graph: bool = True,
        latent_graph_top_k: int = 4, num_relations: int = NUM_RELATIONS, hybrid_alpha_init: float = 0.0) -> None:
        super().__init__()
        self.default_ablation_mode = ablation_mode

        # Process text, tabular metadata, and graph topology through highly specialized independent encoders 
        # before late-stage fusion. Prevent the "modality drowning" effect where high-dimensional text embeddings 
        # overshadow sparse categorical metadata early in the network.
        self.text_encoder = TextEncoder(
            model_name, text_dim, peft_mode, lora_r, lora_alpha, lora_dropout,
            peft_target_modules, gradient_checkpointing, freeze_backbone_until_layer)

        self.metadata_encoder = MetadataEncoder(
            num_venues, num_publishers, num_authors,
            metadata_embedding_dim, metadata_cross_layers, metadata_dim)

        self.citation_encoder = CitationGraphTransformer(
            text_dim, citation_dim, metadata_dim, citation_heads, citation_layers, citation_ff_dim,
            selector_hidden_dim, selector_top_k, max_context_size, citation_dropout,
            hop_profile_dim=hop_profile_dim, spectral_dim=spectral_dim,
            use_latent_graph=use_latent_graph, latent_graph_top_k=latent_graph_top_k,
            num_relations=num_relations, hybrid_alpha_init=hybrid_alpha_init)

        self.fusion = MultimodalFusion(text_dim, metadata_dim, citation_dim, fusion_dim, fusion_modality_dropout)
        self.classifier = NormalizedCosineClassifier(fusion_dim, num_classes, classifier_scale)

    def ablation_study(self, h_text: Tensor, h_meta: Tensor, h_citation: Tensor, mode: str) -> TensorTriplet:
        """Dynamically disables specific modalities based on the requested ablation mode."""
        # DESIGN DECISION: Zero-Masked Ablation
        # Instead of dynamically altering the graph architecture (which breaks batching and 
        # pre-trained dimension sizes), execute the full forward pass and hard-zero the 
        # output tensors of the ablated modalities. The downstream Gated Residual Network (GRN) 
        # natively handles zero-vectors, perfectly simulating a missing modality.
        keep = ABLATION_MODES.get(mode, ABLATION_MODES["full"])
        tensors = (h_text, h_meta, h_citation)
        names = ("text", "metadata", "citation")
        return tuple(t if n in keep else t.new_zeros(t.shape) for n, t in zip(names, tensors))

    def encode_modalities(
        self, input_ids: Tensor, attention_mask: Tensor, venue_ids: Tensor, publisher_ids: Tensor,
        author_ids: Tensor, years: Tensor, context_input_ids: Tensor, context_attention_mask: Tensor,
        context_mask: Tensor, context_edge_types: Tensor, context_year_deltas: Tensor, context_scores: Tensor,
        context_hop_profiles: Tensor | None = None, context_spectral: Tensor | None = None,
        context_venue_ids: Tensor | None = None, context_publisher_ids: Tensor | None = None,
        context_years: Tensor | None = None, ablation_mode: str | None = None) -> TensorTriplet:
        """Executes the specialized encoders and prepares the representations for fusion."""
        
        # 1. Base Document Representations
        h_text = self.text_encoder(input_ids, attention_mask)
        h_meta = self.metadata_encoder(venue_ids, publisher_ids, author_ids, years)
        
        # 2. Graph Context Textual Encoding (Batched)
        candidate_embeddings = self.citation_encoder.encode_candidates(context_input_ids, context_attention_mask, self.text_encoder)

        # 3. Graph Context Metadata Encoding
        context_meta = None
        if context_venue_ids is not None and context_publisher_ids is not None and context_years is not None:
            # Don't cache context author lists because they explode memory requirements.
            # Instead, pass an all-zero author tensor. Because `MetadataEncoder` uses 
            # `padding_idx=0` and a masked mean, these dummy authors mathematically 
            # vanish, allowing the network to encode context venue/publisher seamlessly.
            fake_authors = context_venue_ids.new_zeros(context_venue_ids.size(0), context_venue_ids.size(1), author_ids.size(1))
            flat_meta = self.metadata_encoder(
                context_venue_ids.reshape(-1),
                context_publisher_ids.reshape(-1),
                fake_authors.reshape(-1, fake_authors.size(-1)),
                context_years.reshape(-1, 1))
            context_meta = flat_meta.reshape(context_venue_ids.size(0), context_venue_ids.size(1), -1)

        # 4. Relational Graph Aggregation
        h_citation = self.citation_encoder(
            h_text, h_meta, candidate_embeddings, context_mask,
            context_edge_types, context_year_deltas, context_scores,
            context_hop_profiles, context_spectral, context_meta, context_years)

        return self.ablation_study(h_text, h_meta, h_citation, ablation_mode or self.default_ablation_mode)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, venue_ids: Tensor, publisher_ids: Tensor,
        author_ids: Tensor, years: Tensor, context_input_ids: Tensor, context_attention_mask: Tensor,
        context_mask: Tensor, context_edge_types: Tensor, context_year_deltas: Tensor, context_scores: Tensor,
        context_hop_profiles: Tensor | None = None, context_spectral: Tensor | None = None,
        context_venue_ids: Tensor | None = None, context_publisher_ids: Tensor | None = None,
        context_years: Tensor | None = None, ablation_mode: str | None = None, return_parts: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Executes the full forward pass returning fused embeddings, hypersphere logits, and probabilities."""
        
        h_text, h_meta, h_citation = self.encode_modalities(
            input_ids, attention_mask, venue_ids, publisher_ids, author_ids, years,
            context_input_ids, context_attention_mask, context_mask, context_edge_types, 
            context_year_deltas, context_scores, context_hop_profiles=context_hop_profiles, 
            context_spectral=context_spectral, context_venue_ids=context_venue_ids, 
            context_publisher_ids=context_publisher_ids, context_years=context_years, 
            ablation_mode=ablation_mode)

        embeddings = self.fusion(h_text, h_meta, h_citation)
        logits, probabilities = self.classifier(embeddings)

        if return_parts:
            return embeddings, logits, probabilities, {"text": h_text, "metadata": h_meta, "citation": h_citation}
        return embeddings, logits, probabilities

    def get_embeddings(self, ablation_mode: str | None = None, **batch: Tensor) -> Tensor:
        """Utility for cleanly extracting representations during self-supervised learning."""
        return self.forward(ablation_mode=ablation_mode, **batch)[0]
