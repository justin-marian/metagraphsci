import math
import torch
import torch.nn as nn
from torch import Tensor


class DeepCrossNetwork(nn.Module):
    """Models explicit polynomial feature interactions between metadata fields."""

    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(num_layers)])
        self.biases  = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

        # Each layer scales x0 by that factor, so after 3 layers the norm
        # explodes by ~16^3 = 4 096× (measured: 16 → 39 222).  Under fp16
        # this immediately overflows to NaN, poisoning every metadata gradient.
        # Fix: divide by sqrt(d) so each cross product stays O(1).
        for w in self.weights:
            nn.init.normal_(w, mean=0.0, std=1.0 / math.sqrt(input_dim))

    def forward(self, x0: Tensor) -> Tensor:
        x = x0
        for w, b in zip(self.weights, self.biases):
            x = x0 * torch.sum(x * w, dim=1, keepdim=True) + b + x
        return x


class MetadataEncoder(nn.Module):
    """Maps discrete publication metadata into a dense continuous space."""

    def __init__(
        self, num_venues: int, num_publishers: int, num_authors: int,
        embedding_dim: int, cross_layers: int, output_dim: int) -> None:
        super().__init__()
        metadata_feature_dim = embedding_dim * 4

        self.venue     = nn.Embedding(num_venues, embedding_dim)
        self.publisher = nn.Embedding(num_publishers, embedding_dim)
        # padding_idx=0 → zero-vector for empty author slots (masked mean below)
        self.author = nn.Embedding(num_authors, embedding_dim, padding_idx=0)
        # Single-scalar year → same embedding_dim space so DCN treats it fairly
        self.year   = nn.Linear(1, embedding_dim)

        self.cross = DeepCrossNetwork(metadata_feature_dim, cross_layers)
        self.projection = nn.Sequential(
            nn.Linear(metadata_feature_dim, output_dim),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim))

    def forward(self, venue_ids: Tensor, publisher_ids: Tensor,
                author_ids: Tensor, years: Tensor) -> Tensor:
        years = years.float().view(-1, 1)
        author_embeddings = self.author(author_ids)
        # Masked mean pooling — stable regardless of author-list length
        author_mask = (author_ids != 0).float().unsqueeze(-1)
        author_mean = (
            (author_embeddings * author_mask).sum(dim=1)
            / author_mask.sum(dim=1).clamp_min(1.0))
        features = torch.cat(
            [self.venue(venue_ids), self.publisher(publisher_ids),
            author_mean, self.year(years)], dim=1)
        return self.projection(self.cross(features))
