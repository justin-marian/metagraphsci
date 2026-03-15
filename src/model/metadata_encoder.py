import torch
import torch.nn as nn
from torch import Tensor


class DeepCrossNetwork(nn.Module):
    """Models explicit polynomial feature interactions between metadata fields."""

    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        # Explicit Feature Crossing
        # Standard MLPs are notoriously inefficient at learning explicit multiplicative 
        # feature interactions. The Deep Cross Network explicitly computes bounded 
        # polynomial interactions at each layer without the combinatorial parameter 
        # explosion of full quadratic cross-products.
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim)) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

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
        
        self.venue = nn.Embedding(num_venues, embedding_dim)
        self.publisher = nn.Embedding(num_publishers, embedding_dim)
        # Variable-Length Set Encoding
        # Papers have a variable number of authors. By setting `padding_idx=0`, 
        # the network learns to emit strict zero-vectors for empty padding slots. 
        self.author = nn.Embedding(num_authors, embedding_dim, padding_idx=0)
        
        # Continuous Variable Elevation
        # The normalized publication year is a single scalar. If concatenated 
        # directly with 64-dim embeddings, it gets drowned out. A Linear layer 
        # elevates this scalar to the same `embedding_dim` vector space so the DCN 
        # treats temporal features with equal mathematical weight.
        self.year = nn.Linear(1, embedding_dim)
        self.cross = DeepCrossNetwork(metadata_feature_dim, cross_layers)
        # Projects the high-dimensional crossed features down to the final target dimension
        self.projection = nn.Sequential(
            nn.Linear(metadata_feature_dim, output_dim),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim))

    def forward(self, venue_ids: Tensor, publisher_ids: Tensor, author_ids: Tensor, years: Tensor) -> Tensor:
        years = years.float().view(-1, 1)
        author_embeddings = self.author(author_ids)
        # Masked Mean Pooling
        # Instead of flattening the author embeddings (which would break on variable padding) 
        # or summing them (which artificially inflates the magnitude for papers with 50 authors), 
        # calculate a masked mean. This gives a stable, fixed-size representation of the authorship team's joint latent space.
        author_mask = (author_ids != 0).float().unsqueeze(-1)
        author_mean = (author_embeddings * author_mask).sum(dim=1) / author_mask.sum(dim=1).clamp_min(1.0)
        features = torch.cat([self.venue(venue_ids), self.publisher(publisher_ids), author_mean, self.year(years)], dim=1)
        return self.projection(self.cross(features))
