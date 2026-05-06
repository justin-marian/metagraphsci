"""
Citation Graph Encoder — structural position encoding, relation-aware attention
biasing, local message passing, latent adjacency refinement.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ablation import NUM_RELATIONS


class SignEquivariantPE(nn.Module):
    """Encodes spectral structural signals while reducing sign-flip sensitivity."""

    def __init__(self, spectral_dim: int, output_dim: int) -> None:
        super().__init__()
        self.spectral_dim = spectral_dim
        self.mlp = nn.Sequential(
            nn.Linear(spectral_dim, output_dim), nn.GELU(),
            nn.Linear(output_dim, output_dim))

    def forward(self, eigenvecs: Tensor) -> Tensor:
        if self.training:
            signs = (torch.randint(0, 2, eigenvecs.shape,
                    device=eigenvecs.device,
                    dtype=eigenvecs.dtype) * 2.0 - 1.0)
            eigenvecs = eigenvecs * signs
        else:
            eigenvecs = eigenvecs.abs()
        return self.mlp(eigenvecs)


class StructuralPEEncoder(nn.Module):
    """Aggregates structural cues into one positional representation."""

    def __init__(self, token_dim: int,
        hop_profile_dim: int = 2, spectral_dim: int = 0) -> None:
        super().__init__()
        self.hop_profile_dim = max(hop_profile_dim, 0)
        self.spectral_dim    = max(spectral_dim, 0)

        self.edge_emb   = nn.Embedding(4, token_dim)
        self.time_mlp   = nn.Sequential(
            nn.Linear(1, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        self.score_proj = nn.Linear(1, token_dim)

        if self.hop_profile_dim > 0:
            self.hop_proj = nn.Sequential(
                nn.Linear(hop_profile_dim, token_dim), nn.GELU(),
                nn.Linear(token_dim, token_dim))
        if self.spectral_dim > 0:
            self.spectral_enc = SignEquivariantPE(spectral_dim, token_dim)

        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        edge_types: Tensor, year_deltas: Tensor, selector_scores: Tensor,
        hop_profiles: Tensor | None = None,
        spectral_features: Tensor | None = None,
    ) -> Tensor:
        year_feat = torch.exp(-year_deltas.abs().float().unsqueeze(-1) / 5.0)
        pe = (self.edge_emb(edge_types.clamp(0, 3))
            + self.time_mlp(year_feat)
            + self.score_proj(selector_scores.float().unsqueeze(-1)))

        if self.hop_profile_dim > 0 and hop_profiles is not None:
            pe = pe + self.hop_proj(hop_profiles.float())

        # When spectral computation is disabled or fails, features are all-zero, producing a near-zero pe.
        # LayerNorm on a near-zero tensor (variance → 0) produces NaN, which
        # then propagates through all subsequent attention operations.
        # Fix: skip the spectral branch when the feature tensor is trivially empty.
        if (self.spectral_dim > 0
                and spectral_features is not None
                and spectral_features.abs().max() > 1e-6):
            pe = pe + self.spectral_enc(spectral_features.float())

        return self.norm(pe)


class GraphTokenizer(nn.Module):
    """Splits each context node into semantic content and structural position."""

    def __init__(self, text_dim: int, token_dim: int,
        hop_profile_dim: int = 2, spectral_dim: int = 0) -> None:
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, token_dim), nn.LayerNorm(token_dim))
        self.pe_enc = StructuralPEEncoder(token_dim, hop_profile_dim, spectral_dim)

    def forward(
        self,
        embeddings: Tensor, edge_types: Tensor, year_deltas: Tensor,
        selector_scores: Tensor,
        hop_profiles: Tensor | None = None,
        spectral_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return (
            self.text_proj(embeddings),
            self.pe_enc(edge_types, year_deltas, selector_scores,
                        hop_profiles, spectral_features))


class LearnedCitationSelector(nn.Module):
    """Scores candidate context nodes and retains the most informative subset."""

    def __init__(self, text_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_proj     = nn.Linear(text_dim, hidden_dim)
        self.candidate_proj = nn.Linear(text_dim, hidden_dim)
        self.edge_embedding = nn.Embedding(4, hidden_dim)
        self.time_proj      = nn.Linear(1, hidden_dim)
        self.score_proj     = nn.Linear(1, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 1, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(
        self,
        center_text: Tensor, candidate_embeddings: Tensor,
        candidate_mask: Tensor,
        edge_types: Tensor, year_deltas: Tensor,
        cache_scores: Tensor, top_k: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        _, num_candidates, _ = candidate_embeddings.shape
        query    = self.query_proj(center_text).unsqueeze(1).expand(-1, num_candidates, -1)
        cand     = self.candidate_proj(candidate_embeddings)
        edge     = self.edge_embedding(edge_types.clamp(0, 3))
        time     = self.time_proj(year_deltas.float().unsqueeze(-1))
        prior    = self.score_proj(cache_scores.float().unsqueeze(-1))
        semantic = F.cosine_similarity(
            center_text.unsqueeze(1), candidate_embeddings, dim=-1).unsqueeze(-1)

        features = torch.cat([query, cand, edge, time, prior, semantic], dim=-1)
        logits   = self.scorer(features).squeeze(-1)
        logits   = logits.masked_fill(~candidate_mask.bool(), float('-inf'))

        k = max(1, min(int(top_k), num_candidates))
        top_logits, top_idx = logits.topk(k=k, dim=1)
        return top_idx, top_logits, logits


class LatentGraphModule(nn.Module):
    """Learns a sparse latent adjacency among selected context nodes."""

    def __init__(self, token_dim: int, top_k: int = 4) -> None:
        super().__init__()
        self.q_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.k_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.top_k  = int(top_k)
        self.scale  = token_dim ** -0.5

    def forward(self, neighbor_tokens: Tensor, neighbor_mask: Tensor) -> Tensor:
        B, N, _ = neighbor_tokens.shape
        q = self.q_proj(neighbor_tokens)
        k = self.k_proj(neighbor_tokens)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

        invalid = ~neighbor_mask.bool()
        scores = scores.masked_fill(invalid.unsqueeze(1), float('-inf'))
        scores = scores.masked_fill(invalid.unsqueeze(2), float('-inf'))
        scores = scores.masked_fill(
            torch.eye(N, dtype=torch.bool, device=scores.device).unsqueeze(0), float('-inf'))

        if N > 1:
            effective_k  = min(self.top_k, N - 1)
            topk_vals, _ = scores.topk(effective_k, dim=-1)
            threshold    = topk_vals[..., -1:].detach()
            scores       = scores.masked_fill(scores < threshold, float('-inf'))

        latent_adj = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        return latent_adj * neighbor_mask.float().unsqueeze(-1)


class RelationMixer(nn.Module):
    """Per-head learnable mixture over relation channels."""

    def __init__(self, num_relations: int, num_heads: int) -> None:
        super().__init__()
        self.mixing_logits = nn.Parameter(torch.zeros(num_heads, num_relations))

    def forward(self, relation_indicators: Tensor) -> Tensor:
        alpha = torch.softmax(self.mixing_logits, dim=-1)
        return torch.einsum("bijr,hr->bhij", relation_indicators.float(), alpha)


class LocalMPNNBranch(nn.Module):
    """Gated local neighbourhood aggregation."""

    def __init__(self, token_dim: int, dropout: float) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.GELU(), nn.Dropout(dropout))
        self.gate_proj   = nn.Linear(token_dim * 2, token_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, adjacency: Tensor, node_mask: Tensor) -> Tensor:
        adj = adjacency * node_mask.float().unsqueeze(1)
        adj = adj / adj.sum(-1, keepdim=True).clamp_min(1e-9)
        messages = self.message_mlp(tokens)
        agg      = torch.bmm(adj, messages)
        gate     = torch.sigmoid(self.gate_proj(torch.cat([tokens, agg], dim=-1)))
        return self.dropout(gate * agg)


class GPSCitationLayer(nn.Module):
    """Hybrid relational layer: global transformer + local MPNN."""

    def __init__(
        self, token_dim: int, num_heads: int, ff_dim: int,
        dropout: float, num_relations: int,
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError(f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim  = token_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.token_dim = token_dim

        self.norm_attn   = nn.LayerNorm(token_dim)
        self.q_proj      = nn.Linear(token_dim, token_dim)
        self.k_proj      = nn.Linear(token_dim, token_dim)
        self.v_proj      = nn.Linear(token_dim, token_dim)
        self.out_proj    = nn.Linear(token_dim, token_dim)
        self.pe_to_q     = nn.Linear(token_dim, token_dim, bias=False)
        self.pe_to_k     = nn.Linear(token_dim, token_dim, bias=False)

        self.relation_mixer  = RelationMixer(num_relations, num_heads)
        self.dropout_attn    = nn.Dropout(dropout)
        self.mpnn            = LocalMPNNBranch(token_dim, dropout)
        self.combine_gate    = nn.Sequential(nn.Linear(token_dim * 2, token_dim), nn.Sigmoid())
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
        tokens: Tensor, node_pe: Tensor | None,
        key_padding_mask: Tensor | None,
        relation_indicators: Tensor,
        adjacency: Tensor, node_mask: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        B, T, D = tokens.shape
        residual = tokens
        x = self.norm_attn(tokens)

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if node_pe is not None:
            q = q + self._split_heads(self.pe_to_q(node_pe))
            k = k + self._split_heads(self.pe_to_k(node_pe))

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits + self.relation_mixer(relation_indicators)

        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

        attn_weights = self.dropout_attn(torch.softmax(logits, dim=-1))
        # All-masked rows (paper with zero valid context neighbours)
        # produce all-(-inf) logits, and softmax(-inf, ..., -inf) = NaN.
        # NaNs propagate through matmul into tokens and corrupt the entire batch.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.out_proj(attn_out)

        mpnn_delta = self.mpnn(x, adjacency, node_mask)

        gate     = self.combine_gate(torch.cat([attn_out, mpnn_delta], dim=-1))
        combined = gate * attn_out + (1.0 - gate) * mpnn_delta

        tokens = residual + self.dropout_combine(combined)
        tokens = tokens + self.ff(self.norm_ff(tokens))

        updated_pe: Tensor | None = None
        if node_pe is not None:
            pe_heads   = node_pe.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            pe_agg     = torch.matmul(attn_weights.detach(), pe_heads)
            pe_agg     = pe_agg.transpose(1, 2).contiguous().view(B, T, D)
            updated_pe = F.layer_norm(node_pe + pe_agg, [self.token_dim])

        return tokens, updated_pe


class CitationGraphTransformer(nn.Module):
    """End-to-end citation-context encoder with learned selection and gated pooling."""

    def __init__(
        self,
        text_dim: int, output_dim: int, metadata_dim: int,
        num_heads: int, num_layers: int, ff_dim: int,
        selector_hidden_dim: int, selector_top_k: int, max_context_size: int,
        dropout: float,
        hop_profile_dim: int = 2, spectral_dim: int = 0,
        use_latent_graph: bool = True, latent_graph_top_k: int = 4,
        num_relations: int = NUM_RELATIONS, hybrid_alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.selector_top_k  = int(min(selector_top_k, max_context_size))
        self.hop_profile_dim = int(hop_profile_dim)
        self.spectral_dim    = int(spectral_dim)
        self._num_heads      = int(num_heads)
        self.num_relations   = int(num_relations)

        self.center_proj = nn.Sequential(
            nn.Linear(text_dim, output_dim), nn.LayerNorm(output_dim))
        self.selector    = LearnedCitationSelector(text_dim, selector_hidden_dim, dropout)
        self.tokenizer   = GraphTokenizer(text_dim, output_dim, hop_profile_dim, spectral_dim)
        self.layers      = nn.ModuleList([
            GPSCitationLayer(output_dim, num_heads, ff_dim, dropout, num_relations)
            for _ in range(num_layers)])

        self.meta_proj     = nn.Linear(metadata_dim, output_dim)
        self.use_latent_graph = bool(use_latent_graph)
        if use_latent_graph:
            self.latent_graph = LatentGraphModule(output_dim, top_k=latent_graph_top_k)
            self.hybrid_alpha = nn.Parameter(Tensor([float(hybrid_alpha_init)]))
        else:
            self.latent_graph = None
            self.hybrid_alpha = None

        self.pool_gate   = nn.Linear(output_dim * 2, output_dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(output_dim), nn.Linear(output_dim, output_dim))

    @staticmethod
    def encode_candidates(
        candidate_input_ids: Tensor,
        candidate_attention_mask: Tensor,
        text_encoder: nn.Module,
    ) -> Tensor:
        B, M, S = candidate_input_ids.shape
        flat_ids  = candidate_input_ids.reshape(B * M, S)
        flat_mask = candidate_attention_mask.reshape(B * M, S)
        return text_encoder(flat_ids, flat_mask).reshape(B, M, -1)

    @staticmethod
    def gather_last_dim(values: Tensor, indices: Tensor) -> Tensor:
        return values.gather(1, indices.unsqueeze(-1).expand(-1, -1, values.size(-1)))

    @staticmethod
    def gather_vector(values: Tensor, indices: Tensor) -> Tensor:
        return values.gather(1, indices)

    def build_relation_indicators(
        self,
        selected_edges: Tensor, selected_years: Tensor,
        center_metadata: Tensor, context_meta: Tensor | None,
        latent_adj: Tensor | None, total_tokens: int,
    ) -> Tensor:
        B, K = selected_edges.shape
        indicators = selected_edges.new_zeros(
            B, total_tokens, total_tokens, self.num_relations, dtype=torch.float32)

        edge_norm  = selected_edges.float() / 3.0
        indicators[:, 0, 1:, 0] = edge_norm
        indicators[:, 1:, 0, 0] = edge_norm

        year_prox  = torch.exp(-selected_years.abs().float() / 5.0)
        indicators[:, 0, 1:, 1] = year_prox
        indicators[:, 1:, 0, 1] = year_prox

        if context_meta is not None and self.num_relations >= 3:
            c_proj = F.normalize(self.meta_proj(center_metadata.float()), dim=-1)
            n_proj = F.normalize(
                self.meta_proj(context_meta.float().reshape(-1, context_meta.size(-1)))
                .reshape(B, K, -1), dim=-1)
            meta_compat = (c_proj.unsqueeze(1) * n_proj).sum(-1).clamp(-1.0, 1.0)
            indicators[:, 0, 1:, 2] = meta_compat
            indicators[:, 1:, 0, 2] = meta_compat

        if latent_adj is not None and self.num_relations >= 4:
            indicators[:, 1:, 1:, 3] = latent_adj

        return indicators

    def build_adjacency(self, relation_indicators: Tensor, full_mask: Tensor) -> Tensor:
        adj = relation_indicators.mean(dim=-1)
        m   = full_mask.float()
        return adj * m.unsqueeze(1) * m.unsqueeze(2)

    def forward(
        self,
        center_text: Tensor, center_metadata: Tensor,
        candidate_embeddings: Tensor, candidate_mask: Tensor,
        edge_types: Tensor, year_deltas: Tensor, cache_scores: Tensor,
        hop_profiles: Tensor | None = None,
        spectral_features: Tensor | None = None,
        context_metadata: Tensor | None = None,
    ) -> Tensor:
        B = center_text.size(0)

        top_idx, top_logits, _ = self.selector(
            center_text, candidate_embeddings, candidate_mask,
            edge_types, year_deltas, cache_scores, top_k=self.selector_top_k)

        sel_emb    = self.gather_last_dim(candidate_embeddings, top_idx)
        sel_mask   = self.gather_vector(candidate_mask, top_idx)
        sel_edges  = self.gather_vector(edge_types, top_idx)
        sel_years  = self.gather_vector(year_deltas, top_idx)
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

        neighbor_feats, neighbor_pe = self.tokenizer(
            sel_emb, sel_edges, sel_years, sel_scores, sel_hops, sel_spectral)

        center_feat = self.center_proj(center_text).unsqueeze(1)
        center_pe   = neighbor_pe.new_zeros(B, 1, neighbor_pe.size(-1))

        tokens   = torch.cat([center_feat, neighbor_feats * sel_mask.unsqueeze(-1).float()], dim=1)
        node_pe  = torch.cat([center_pe, neighbor_pe], dim=1)
        full_mask = torch.cat([sel_mask.new_ones(B, 1), sel_mask], dim=1).bool()
        key_padding_mask = ~full_mask

        latent_adj: Tensor | None = None
        if self.use_latent_graph and self.latent_graph is not None and sel_mask.any():
            raw_latent = self.latent_graph(neighbor_feats, sel_mask.bool())
            alpha      = torch.sigmoid(self.hybrid_alpha)
            latent_adj = alpha * raw_latent

        T = tokens.size(1)
        relation_indicators = self.build_relation_indicators(
            sel_edges, sel_years, center_metadata, sel_ctx_meta, latent_adj, T)
        adjacency = self.build_adjacency(relation_indicators, full_mask)

        for layer in self.layers:
            tokens, node_pe = layer(
                tokens, node_pe, key_padding_mask, relation_indicators, adjacency, full_mask)

        center_out   = tokens[:, 0]
        neighbor_sum = (tokens[:, 1:] * sel_mask.unsqueeze(-1).float()).sum(1)
        neighbor_cnt = sel_mask.float().sum(1, keepdim=True).clamp_min(1.0)
        pooled       = neighbor_sum / neighbor_cnt

        gate = torch.sigmoid(self.pool_gate(torch.cat([center_out, pooled], dim=-1)))
        return self.output_proj(center_out + gate * pooled)
