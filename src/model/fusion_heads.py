import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultimodalFusion(nn.Module):
    """Gated residual fusion of text, metadata, and citation streams."""

    def __init__(self, text_dim: int, metadata_dim: int, citation_dim: int,
                fusion_dim: int, modality_dropout: float) -> None:
        super().__init__()
        # Text acts as anchor — never dropped.  Dropping metadata/citation trains
        # the model to work gracefully when those streams are absent at inference.
        self.modality_dropout = float(modality_dropout)
        total_dim = text_dim + metadata_dim + citation_dim
        self.input_proj = nn.Sequential(
            nn.Linear(total_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        self.fuser = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim))
        self.gate = nn.Linear(total_dim, fusion_dim)

    def maybe_drop(self, x: Tensor) -> Tensor:
        """Zeros entire modality vectors for random items in the batch.

        With the old code and p=0.5, training used 2× magnitudes, so at evaluation the
        fusion gate was systematically miscalibrated — the full model scored
        BELOW random while text_only (whose modalities are always zeroed by
        ablation_study) was unaffected.  Fix: binary masking only.
        """
        if self.training and self.modality_dropout > 0.0:
            keep = torch.bernoulli(
                torch.full((x.size(0), 1), 1.0 - self.modality_dropout, device=x.device))
            x = x * keep          # 0 when dropped, unchanged (1×) when kept
        return x

    def forward(self, h_text: Tensor, h_meta: Tensor, h_citation: Tensor) -> Tensor:
        concatenated = torch.cat(
            [h_text, self.maybe_drop(h_meta), self.maybe_drop(h_citation)], dim=1)
        residual = self.input_proj(concatenated)
        mixed    = self.fuser(concatenated)
        gate     = torch.sigmoid(self.gate(concatenated))
        return residual + gate * mixed


class NormalizedCosineClassifier(nn.Module):
    """Cosine-similarity classifier with learnable class prototypes."""

    def __init__(self, input_dim: int, num_classes: int, scale: float) -> None:
        super().__init__()
        # L2-normalise both embeddings and prototypes so logits are bounded to
        # [-scale, +scale].  The scale factor (typically 10–30) sharpens the
        # softmax enough to produce meaningful CE gradients.
        self.scale = float(scale)
        self.class_vectors = nn.Parameter(torch.empty(num_classes, input_dim))
        nn.init.normal_(self.class_vectors, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        nx  = F.normalize(x, p=2, dim=1)
        np_ = F.normalize(self.class_vectors, p=2, dim=1)
        logits = self.scale * (nx @ np_.t())
        return logits, F.softmax(logits, dim=1)
