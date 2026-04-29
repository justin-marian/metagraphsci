"""
Citation Graph Encoder - graph-structured contextual encoder with structural position encoding,
relation-aware attention biasing, local message passing, and latent adjacency refinement.

Key design updates:
    1.  Sign-equivariant spectral encoding
        Handles Laplacian eigenvector sign ambiguity more robustly.
---
    2.  Structural/semantic separation
        Structural cues influence routing, while semantic content remains clean.
---
    3.  Single structural path
        Structural information is injected only through dedicated positional/bias channels.
---
    4.  Local relational propagation
        Adds explicit neighborhood aggregation alongside global interaction.
---
    5.  Layer-specific relation weighting
        Each layer can learn a different preference over relation types.
---
    6.  Head-specific relation weighting
        Different attention heads may emphasize different relational signals.
---
    7.  Unified relation-bias construction
        Relation signals are assembled once per layer in a single consistent path.
---
    8.  Zero-dimension safeguards
        Optional structural components are skipped cleanly when disabled.
---
    9.  Numerically stable masking
        Uses float('-inf') masking, which is safer in mixed precision.
---
    10. NaN protection in latent adjacency
        Fully masked rows are sanitized after normalization.

References:
    Rampasek et al. (2022), "Recipe for a General, Powerful, Scalable Graph Transformer"
    Ostendorff et al. (2022), "Neighborhood Contrastive Learning for Scientific Document Representations"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ablation import NUM_RELATIONS


class SignEquivariantPE(nn.Module):
    """
    Encodes spectral structural signals while reducing sensitivity to eigenvector sign flips.

    Laplacian eigenvectors are defined only up to sign, so two equivalent graph
    decompositions may differ by a global sign inversion. This module makes the
    representation less sensitive to that ambiguity by applying random sign flips
    during training and a canonical absolute-value form at inference.
    """

    def __init__(self, spectral_dim: int, output_dim: int) -> None:
        super().__init__()
        self.spectral_dim = spectral_dim
        self.mlp = nn.Sequential(nn.Linear(spectral_dim, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim))

    def forward(self, eigenvecs: Tensor) -> Tensor:
        """eigenvecs: (B, N, spectral_dim)"""
        if self.training:
            # Random sign inversion exposes the encoder to equivalent spectral
            # configurations so the learned representation becomes less dependent
            # on an arbitrary eigenvector orientation.
            signs = torch.randint(0, 2, eigenvecs.shape, device=eigenvecs.device, dtype=eigenvecs.dtype) * 2.0 - 1.0
            eigenvecs = eigenvecs * signs
        else:
            # A deterministic absolute-value form provides a stable inference-time
            # representation for the same underlying spectral structure.
            eigenvecs = eigenvecs.abs()
        return self.mlp(eigenvecs)


class StructuralPEEncoder(nn.Module):
    """
    Aggregates multiple structural cues into one positional representation.

    The goal is to encode graph-related context separately from semantic content,
    so structural signals can guide interaction patterns without contaminating
    the content representation itself.

    The implementation is guarded so optional components are created only when
    their corresponding dimensions are enabled.
    """

    def __init__(self, token_dim: int, hop_profile_dim: int = 2, spectral_dim: int = 0) -> None:
        super().__init__()
        self.hop_profile_dim = max(hop_profile_dim, 0)
        self.spectral_dim = max(spectral_dim, 0)

        self.edge_emb = nn.Embedding(4, token_dim)
        self.time_mlp = nn.Sequential(nn.Linear(1, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        self.score_proj = nn.Linear(1, token_dim)

        if self.hop_profile_dim > 0:
            self.hop_proj = nn.Sequential(nn.Linear(hop_profile_dim, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))

        if self.spectral_dim > 0:
            self.spectral_enc = SignEquivariantPE(spectral_dim, token_dim)

        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        edge_types: Tensor, year_deltas: Tensor, selector_scores: Tensor,
        hop_profiles: Tensor | None = None, spectral_features: Tensor | None = None
    ) -> Tensor:
        # Convert temporal distance into a smooth decay signal so nearby years
        # remain more similar than distant ones.
        year_feat = torch.exp(-year_deltas.abs().float().unsqueeze(-1) / 5.0)

        # Core structural encoding combines relation type, temporal proximity,
        # and selection confidence into one positional descriptor.
        pe = (
            self.edge_emb(edge_types.clamp(0, 3)) +
            self.time_mlp(year_feat) +
            self.score_proj(selector_scores.float().unsqueeze(-1)))

        # Optional hop-profile information adds multi-hop structural context
        # when such descriptors are available.
        if self.hop_profile_dim > 0 and hop_profiles is not None:
            pe = pe + self.hop_proj(hop_profiles.float())

        # Optional spectral features inject global graph structure in a form
        # that is more robust to sign ambiguity.
        if self.spectral_dim > 0 and spectral_features is not None:
            pe = pe + self.spectral_enc(spectral_features.float())

        # Final normalization keeps the combined structural representation well-scaled
        # before it is reused inside attention.
        return self.norm(pe)


class GraphTokenizer(nn.Module):
    """
    Splits each contextual node into semantic content and structural position.

    The semantic branch is intended to carry document meaning, while the structural
    branch captures graph-derived cues that influence interaction patterns.
    Keeping them separate makes it easier to control where structure enters the model.
    """

    def __init__(
        self, text_dim: int, token_dim: int,
        hop_profile_dim: int = 2, spectral_dim: int = 0) -> None:
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, token_dim), nn.LayerNorm(token_dim))
        self.pe_enc = StructuralPEEncoder(token_dim, hop_profile_dim, spectral_dim)

    def forward(
        self,
        embeddings: Tensor, edge_types: Tensor, year_deltas: Tensor, selector_scores: Tensor,
        hop_profiles: Tensor | None = None, spectral_features: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Returns semantic features and structural encodings, each of shape (B, N, token_dim)."""
        return (
            self.text_proj(embeddings),
            self.pe_enc(edge_types, year_deltas, selector_scores, hop_profiles, spectral_features))


class LearnedCitationSelector(nn.Module):
    """
    Scores candidate context nodes and retains only the most informative subset.

    This acts as a learned context bottleneck: a larger pool is scored first,
    then reduced to a smaller set before deeper relational processing.
    """

    def __init__(self, text_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
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
        self,
        center_text: Tensor, candidate_embeddings: Tensor, candidate_mask: Tensor,
        edge_types: Tensor, year_deltas: Tensor, cache_scores: Tensor, top_k: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        _, num_candidates, _ = candidate_embeddings.shape

        # Build a joint scoring representation that mixes the central item,
        # candidate semantics, relation type, temporal signal, prior score,
        # and direct semantic similarity.
        query = self.query_proj(center_text).unsqueeze(1).expand(-1, num_candidates, -1)
        cand = self.candidate_proj(candidate_embeddings)
        edge = self.edge_embedding(edge_types.clamp(0, 3))
        time = self.time_proj(year_deltas.float().unsqueeze(-1))
        prior = self.score_proj(cache_scores.float().unsqueeze(-1))
        semantic = F.cosine_similarity(
            center_text.unsqueeze(1), candidate_embeddings, dim=-1).unsqueeze(-1)

        features = torch.cat([query, cand, edge, time, prior, semantic], dim=-1)
        logits = self.scorer(features).squeeze(-1)

        # Invalid candidates are removed directly at the logit level using
        # negative infinity masking for stable downstream top-k selection.
        logits = logits.masked_fill(~candidate_mask.bool(), float('-inf'))

        # Retain the strongest subset while guaranteeing at least one candidate
        # survives when the input is non-empty.
        k = max(1, min(int(top_k), num_candidates))
        top_logits, top_idx = logits.topk(k=k, dim=1)
        return top_idx, top_logits, logits


class LatentGraphModule(nn.Module):
    """
    Learns a sparse latent adjacency among the selected context nodes.

    This module complements explicit graph structure with a learned similarity-based
    neighborhood, allowing the model to recover potentially useful interactions
    that were not provided directly by the input graph.
    """

    def __init__(self, token_dim: int, top_k: int = 4) -> None:
        super().__init__()
        self.q_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.k_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.top_k = int(top_k)
        self.scale = token_dim ** -0.5

    def forward(self, neighbor_tokens: Tensor, neighbor_mask: Tensor) -> Tensor:
        B, N, _ = neighbor_tokens.shape

        # Project node representations into a similarity space and compute
        # pairwise compatibility scores.
        q = self.q_proj(neighbor_tokens)
        k = self.k_proj(neighbor_tokens)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

        invalid = ~neighbor_mask.bool()

        # Remove padded positions and self-connections so only valid cross-node
        # links participate in latent graph construction.
        scores = scores.masked_fill(invalid.unsqueeze(1), float('-inf'))
        scores = scores.masked_fill(invalid.unsqueeze(2), float('-inf'))
        scores = scores.masked_fill(torch.eye(N, dtype=torch.bool, device=scores.device).unsqueeze(0), float('-inf'))

        # Keep only the strongest local interactions per row to encourage a sparse,
        # interpretable latent adjacency instead of a fully dense one.
        if N > 1:
            effective_k = min(self.top_k, N - 1)
            topk_vals, _ = scores.topk(effective_k, dim=-1)
            threshold = topk_vals[..., -1:].detach()
            scores = scores.masked_fill(scores < threshold, float('-inf'))

        # Softmax over fully masked rows can produce NaNs, so the result is cleaned
        # before being used downstream.
        latent_adj = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        return latent_adj * neighbor_mask.float().unsqueeze(-1)


class RelationMixer(nn.Module):
    """
    Learns how strongly each relation channel should influence attention.

    Relation signals are not assumed to contribute equally. This module learns
    a separate mixture over relation types for each attention head, allowing
    distinct heads to specialize to different structural cues.
    """

    def __init__(self, num_relations: int, num_heads: int) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.mixing_logits = nn.Parameter(torch.zeros(num_heads, num_relations))

    def forward(self, relation_indicators: Tensor) -> Tensor:
        """
        relation_indicators : (B, T, T, R)
        Returns             : (B, H, T, T) additive attention bias
        """
        # Convert per-head relation preferences into a convex mixture over
        # available relation channels, then project them into attention-bias space.
        alpha = torch.softmax(self.mixing_logits, dim=-1)
        return torch.einsum("bijr,hr->bhij", relation_indicators.float(), alpha)


class LocalMPNNBranch(nn.Module):
    """
    Performs gated local neighborhood aggregation.

    This branch emphasizes nearby relational evidence through explicit message
    passing and complements the broader interaction mechanism in the main layer.
    """

    def __init__(self, token_dim: int, dropout: float) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.GELU(), nn.Dropout(dropout))
        self.gate_proj = nn.Linear(token_dim * 2, token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, adjacency: Tensor, node_mask: Tensor) -> Tensor:
        # Apply validity masking to the adjacency, then row-normalize so each node
        # aggregates a stable weighted neighborhood summary.
        adj = adjacency * node_mask.float().unsqueeze(1)
        adj = adj / adj.sum(-1, keepdim=True).clamp_min(1e-9)

        # Transform node states into messages and accumulate them from neighbors.
        messages = self.message_mlp(tokens)
        agg = torch.bmm(adj, messages)

        # Learn how much of the aggregated local signal should be injected back
        # into each token representation.
        gate = torch.sigmoid(self.gate_proj(torch.cat([tokens, agg], dim=-1)))
        return self.dropout(gate * agg)


class GPSCitationLayer(nn.Module):
    """
    Hybrid relational layer combining global interaction and local propagation.

    The layer processes tokens through two complementary paths:
        - a global interaction path with relation-aware attention biasing
        - a local neighborhood aggregation path with learned gating

    Structural encodings are injected only into the routing path, while semantic
    content remains separate in the value stream. The two branches are fused
    adaptively and followed by a position-wise feed-forward update.
    """

    def __init__(
        self, token_dim: int, num_heads: int, ff_dim: int,
        dropout: float, num_relations: int,
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.token_dim = token_dim

        self.norm_attn = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.out_proj = nn.Linear(token_dim, token_dim)

        # Structural encodings are mapped only into the routing components,
        # keeping content aggregation separate from positional structure.
        self.pe_to_q = nn.Linear(token_dim, token_dim, bias=False)
        self.pe_to_k = nn.Linear(token_dim, token_dim, bias=False)

        self.relation_mixer = RelationMixer(num_relations, num_heads)
        self.dropout_attn = nn.Dropout(dropout)

        self.mpnn = LocalMPNNBranch(token_dim, dropout)

        self.combine_gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim), nn.Sigmoid())
        self.dropout_combine = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(token_dim)
        self.ff = nn.Sequential(
            nn.Linear(token_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, token_dim), nn.Dropout(dropout))

    def _split_heads(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        tokens: Tensor,
        node_pe: Tensor | None,
        key_padding_mask: Tensor | None,
        relation_indicators: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        B, T, D = tokens.shape
        residual = tokens
        x = self.norm_attn(tokens)

        # Project normalized tokens into query, key, and value subspaces.
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # Structural encodings affect how interactions are routed, but do not alter
        # the semantic content being aggregated.
        if node_pe is not None:
            q = q + self._split_heads(self.pe_to_q(node_pe))
            k = k + self._split_heads(self.pe_to_k(node_pe))

        # Global interaction scores combine content similarity with learned
        # relation-aware bias terms.
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits + self.relation_mixer(relation_indicators)

        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

        attn_weights = self.dropout_attn(torch.softmax(logits, dim=-1))
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.out_proj(attn_out)

        # Local propagation provides complementary short-range relational evidence.
        mpnn_delta = self.mpnn(x, adjacency, node_mask)

        # Learn an elementwise interpolation between the global and local branches
        # rather than assuming one is always superior.
        gate = self.combine_gate(torch.cat([attn_out, mpnn_delta], dim=-1))
        combined = gate * attn_out + (1.0 - gate) * mpnn_delta

        # Apply residual fusion followed by a position-wise feed-forward update.
        tokens = residual + self.dropout_combine(combined)
        tokens = tokens + self.ff(self.norm_ff(tokens))

        # Structural encodings are also updated across layers so routing-related
        # context can evolve together with the token representations.
        updated_pe: Tensor | None = None
        if node_pe is not None:
            pe_heads = node_pe.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            pe_agg = torch.matmul(attn_weights.detach(), pe_heads)
            pe_agg = pe_agg.transpose(1, 2).contiguous().view(B, T, D)
            updated_pe = F.layer_norm(node_pe + pe_agg, [self.token_dim])

        return tokens, updated_pe


class CitationGraphTransformer(nn.Module):
    """
    End-to-end citation-context encoder with learned selection, structural encoding,
    hybrid relational processing, and gated pooling.

    The overall pipeline first selects the most useful contextual nodes, then
    constructs semantic and structural token views, applies several relational
    processing layers, and finally produces a single fused representation.
    """

    def __init__(
        self,
        text_dim: int, output_dim: int, metadata_dim: int,
        num_heads: int, num_layers: int, ff_dim: int,
        selector_hidden_dim: int, selector_top_k: int, max_context_size: int,
        dropout: float,
        hop_profile_dim: int = 2, spectral_dim: int = 0,
        use_latent_graph: bool = True, latent_graph_top_k: int = 4,
        num_relations: int = NUM_RELATIONS, hybrid_alpha_init: float = 0.0
    ) -> None:
        super().__init__()

        self.selector_top_k = int(min(selector_top_k, max_context_size))
        self.hop_profile_dim = int(hop_profile_dim)
        self.spectral_dim = int(spectral_dim)
        self._num_heads = int(num_heads)
        self.num_relations = int(num_relations)

        self.center_proj = nn.Sequential(
            nn.Linear(text_dim, output_dim), nn.LayerNorm(output_dim))

        self.selector = LearnedCitationSelector(text_dim, selector_hidden_dim, dropout)
        self.tokenizer = GraphTokenizer(text_dim, output_dim, hop_profile_dim, spectral_dim)

        self.layers = nn.ModuleList([
            GPSCitationLayer(output_dim, num_heads, ff_dim, dropout, num_relations)
            for _ in range(num_layers)
        ])

        # Metadata from the central item and contextual items is projected into
        # the same latent space before compatibility is measured.
        self.meta_proj = nn.Linear(metadata_dim, output_dim)

        self.use_latent_graph = bool(use_latent_graph)
        if use_latent_graph:
            self.latent_graph = LatentGraphModule(output_dim, top_k=latent_graph_top_k)

            # The learned latent adjacency starts with a modest contribution so
            # explicit structure can dominate initially and the inferred graph
            # can grow in influence if it proves useful.
            self.hybrid_alpha = nn.Parameter(Tensor([float(hybrid_alpha_init)]))
        else:
            self.latent_graph = None
            self.hybrid_alpha = None

        self.pool_gate = nn.Linear(output_dim * 2, output_dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(output_dim), nn.Linear(output_dim, output_dim))

    @staticmethod
    def encode_candidates(
        candidate_input_ids: Tensor,
        candidate_attention_mask: Tensor,
        text_encoder: nn.Module
    ) -> Tensor:
        # Flatten the candidate axis so the external text encoder can process
        # all candidates in one batched call, then restore the original layout.
        B, M, S = candidate_input_ids.shape
        flat_ids = candidate_input_ids.reshape(B * M, S)
        flat_mask = candidate_attention_mask.reshape(B * M, S)
        return text_encoder(flat_ids, flat_mask).reshape(B, M, -1)

    @staticmethod
    def gather_last_dim(values: Tensor, indices: Tensor) -> Tensor:
        # Gather selected contextual features while preserving the trailing feature dimension.
        return values.gather(1, indices.unsqueeze(-1).expand(-1, -1, values.size(-1)))

    @staticmethod
    def gather_vector(values: Tensor, indices: Tensor) -> Tensor:
        # Gather selected one-dimensional contextual attributes such as masks,
        # edge types, or year offsets.
        return values.gather(1, indices)

    def build_relation_indicators(
        self,
        selected_edges: Tensor, selected_years: Tensor,
        center_metadata: Tensor, context_meta: Tensor | None,
        latent_adj: Tensor | None, total_tokens: int
    ) -> Tensor:
        """
        Builds a unified relational descriptor tensor used to bias interactions.

        Relation channels:
            0 - explicit citation relation strength
            1 - temporal proximity
            2 - metadata compatibility
            3 - inferred latent adjacency
        """
        B, K = selected_edges.shape
        indicators = selected_edges.new_zeros(
            B, total_tokens, total_tokens, self.num_relations, dtype=torch.float32)

        # Explicit edge type is normalized into a compact relation-strength signal.
        edge_norm = selected_edges.float() / 3.0
        indicators[:, 0, 1:, 0] = edge_norm
        indicators[:, 1:, 0, 0] = edge_norm

        # Temporal distance is converted into a smooth proximity kernel.
        year_prox = torch.exp(-selected_years.abs().float() / 5.0)
        indicators[:, 0, 1:, 1] = year_prox
        indicators[:, 1:, 0, 1] = year_prox

        # Metadata compatibility is computed only when contextual metadata is available.
        if context_meta is not None and self.num_relations >= 3:
            c_proj = F.normalize(self.meta_proj(center_metadata.float()), dim=-1)
            n_proj = F.normalize(
                self.meta_proj(context_meta.float().reshape(-1, context_meta.size(-1)))
                .reshape(B, K, -1), dim=-1)
            meta_compat = (c_proj.unsqueeze(1) * n_proj).sum(-1).clamp(-1.0, 1.0)
            indicators[:, 0, 1:, 2] = meta_compat
            indicators[:, 1:, 0, 2] = meta_compat

        # A learned latent graph can be injected as an additional relation channel.
        if latent_adj is not None and self.num_relations >= 4:
            indicators[:, 1:, 1:, 3] = latent_adj

        return indicators

    def build_adjacency(self, relation_indicators: Tensor, full_mask: Tensor) -> Tensor:
        """Constructs a soft adjacency matrix for local propagation."""
        # Average across relation channels to obtain one soft connectivity matrix,
        # then suppress padded rows and columns.
        adj = relation_indicators.mean(dim=-1)
        m = full_mask.float()
        return adj * m.unsqueeze(1) * m.unsqueeze(2)

    def forward(
        self,
        center_text: Tensor, center_metadata: Tensor,
        candidate_embeddings: Tensor, candidate_mask: Tensor,
        edge_types: Tensor, year_deltas: Tensor, cache_scores: Tensor, hop_profiles: Tensor | None = None,
        spectral_features: Tensor | None = None, context_metadata: Tensor | None = None
    ) -> Tensor:
        B = center_text.size(0)

        # Score the available context and keep only the strongest subset for
        # deeper relational processing.
        top_idx, top_logits, _ = self.selector(
            center_text, candidate_embeddings, candidate_mask,
            edge_types, year_deltas, cache_scores, top_k=self.selector_top_k)

        # Gather the selected contextual features and their associated structural attributes.
        sel_emb = self.gather_last_dim(candidate_embeddings, top_idx)
        sel_mask = self.gather_vector(candidate_mask, top_idx)
        sel_edges = self.gather_vector(edge_types, top_idx)
        sel_years = self.gather_vector(year_deltas, top_idx)
        sel_scores = torch.sigmoid(top_logits)

        sel_hops = (
            self.gather_last_dim(hop_profiles, top_idx)
            if hop_profiles is not None and hop_profiles.size(-1) > 0 else None)
        sel_spectral = (
            self.gather_last_dim(spectral_features, top_idx)
            if spectral_features is not None and spectral_features.size(-1) > 0 else None)
        sel_ctx_meta = (
            self.gather_last_dim(context_metadata, top_idx)
            if context_metadata is not None else None)

        # Split selected context into semantic token features and structural encodings.
        neighbor_feats, neighbor_pe = self.tokenizer(
            sel_emb, sel_edges, sel_years, sel_scores, sel_hops, sel_spectral)

        # The central document is prepended as token 0 and receives a zero-initialized
        # structural encoding so all tokens share a common layout.
        center_feat = self.center_proj(center_text).unsqueeze(1)
        center_pe = neighbor_pe.new_zeros(B, 1, neighbor_pe.size(-1))

        tokens = torch.cat(
            [center_feat, neighbor_feats * sel_mask.unsqueeze(-1).float()], dim=1)
        node_pe = torch.cat([center_pe, neighbor_pe], dim=1)

        full_mask = torch.cat([sel_mask.new_ones(B, 1), sel_mask], dim=1).bool()
        key_padding_mask = ~full_mask

        latent_adj: Tensor | None = None
        if self.use_latent_graph and self.latent_graph is not None and sel_mask.any():
            # Infer an additional soft neighborhood from contextual similarity and
            # scale its contribution with a learned mixing factor.
            raw_latent = self.latent_graph(neighbor_feats, sel_mask.bool())
            alpha = torch.sigmoid(self.hybrid_alpha)
            latent_adj = alpha * raw_latent

        T = tokens.size(1)

        # Assemble all explicit and inferred relation signals into one tensor
        # that will drive both global biasing and local propagation.
        relation_indicators = self.build_relation_indicators(
            sel_edges, sel_years, center_metadata, sel_ctx_meta, latent_adj, T)

        adjacency = self.build_adjacency(relation_indicators, full_mask)

        # Repeated relational processing updates both token states and structural
        # routing encodings across layers.
        for layer in self.layers:
            tokens, node_pe = layer(
                tokens, node_pe, key_padding_mask,
                relation_indicators, adjacency, full_mask)

        # The final representation combines the updated central token with a
        # masked pooled summary over contextual neighbors.
        center_out = tokens[:, 0]
        neighbor_sum = (tokens[:, 1:] * sel_mask.unsqueeze(-1).float()).sum(1)
        neighbor_cnt = sel_mask.float().sum(1, keepdim=True).clamp_min(1.0)
        pooled = neighbor_sum / neighbor_cnt

        # Gated pooling lets the model decide how much context should refine
        # the central representation before the final projection.
        gate = torch.sigmoid(self.pool_gate(torch.cat([center_out, pooled], dim=-1)))
        return self.output_proj(center_out + gate * pooled)
