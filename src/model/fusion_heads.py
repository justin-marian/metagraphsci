import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultimodalFusion(nn.Module):
    """Gated residual fusion of text, metadata, and citation streams."""

    def __init__(self, text_dim: int, metadata_dim: int, citation_dim: int, fusion_dim: int, modality_dropout: float) -> None:
        super().__init__()
        # Notice that `h_text` is never subjected to modality dropout in the forward pass. 
        # Text acts as the foundational "anchor" modality. By randomly dropping the metadata 
        # and citation vectors during training, force the text encoder to become highly 
        # robust, while also teaching the network how to gracefully handle missing graph 
        # structures during inference (e.g., a brand new paper with 0 citations).
        self.modality_dropout = float(modality_dropout)
        total_dim = text_dim + metadata_dim + citation_dim
        self.input_proj = nn.Sequential(nn.Linear(total_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        # Concatenation + linear projection often fails to capture deep, non-linear 
        # interactions between modalities. Here, `fuser` learns the complex interactions, 
        # while the `gate` (Sigmoid) dynamically decides how much of this new non-linear 
        # mixture should be added back into the preserved linear projection.
        self.fuser = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim))
        self.gate = nn.Linear(total_dim, fusion_dim)

    def maybe_drop(self, x: Tensor) -> Tensor:
        """Drops entire modality vectors for random items in the batch."""
        # Generates a single boolean per batch item `(x.size(0), 1)`, zeroing out the 
        # entire modality embedding for that item, perfectly simulating missing data.
        if self.training and self.modality_dropout > 0.0:
            keep = torch.bernoulli(torch.full((x.size(0), 1), 1.0 - self.modality_dropout, device=x.device))
            x = x * keep / max(1.0 - self.modality_dropout, 1e-9)
        return x

    def forward(self, h_text: Tensor, h_meta: Tensor, h_citation: Tensor) -> Tensor:
        concatenated = torch.cat([h_text, self.maybe_drop(h_meta), self.maybe_drop(h_citation)], dim=1)
        residual = self.input_proj(concatenated)
        mixed = self.fuser(concatenated)
        gate = torch.sigmoid(self.gate(concatenated))
        return residual + gate * mixed


class NormalizedCosineClassifier(nn.Module):
    """Cosine-similarity classifier with learnable class prototypes."""

    def __init__(self, input_dim: int, num_classes: int, scale: float) -> None:
        super().__init__()
        # Standard classifiers (Dot Product) allow logits to grow unbounded, which 
        # often destabilizes training or causes the model to optimize for magnitude rather 
        # than semantic direction. By L2-normalizing both the input embeddings and the 
        # class weights, constrain the embeddings to a unit hypersphere. The weights 
        # become pure "directional prototypes" for each class.
        self.scale = float(scale)
        self.class_vectors = nn.Parameter(torch.empty(num_classes, input_dim))
        nn.init.normal_(self.class_vectors, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        nx = F.normalize(x, p=2, dim=1)
        np_ = F.normalize(self.class_vectors, p=2, dim=1)
        # Cosine similarity is bounded strictly between [-1, 1]. Passing values this small 
        # into a Softmax results in a very flat probability distribution, crippling the 
        # CE gradient. Multiplying by a learnable or fixed `scale` (often > 10.0) 
        # artificially sharpens the logits, restoring standard gradient flow.
        logits = self.scale * (nx @ np_.t())
        return logits, F.softmax(logits, dim=1)
