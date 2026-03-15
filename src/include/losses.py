import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NeighborhoodAwareContrastiveLoss(nn.Module):
    """
    MCNA contrastive loss with neighborhood masking and metadata-aware downweighting.
    
    Encourage embeddings of positive pairs to be close while pushing apart negative pairs. 
    It modifies standard contrastive loss by ignoring known graph neighbors 
    (preventing false negatives) and applying a softer penalty to documents 
    that share similar metadata (like venue or publisher).
    """

    def __init__(self, temperature: float = 0.07, metadata_negative_weight: float = 0.25) -> None:
        # Contrastive Temperature and Stable Log Scaling
        # The temperature controls how sharply similarity scores are separated.
        # Lower values make the softmax more selective
        super().__init__()
        self.temperature = float(temperature)
        self.metadata_negative_weight = float(metadata_negative_weight)
        self.eps = float(1e-9)

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        batch_doc_ids: Tensor,
        batch_neighborhoods: list[set[int]],
        metadata_affinity: Tensor | None = None,
        positive_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute the neighborhood-aware contrastive objective for a mini-batch."""
        # Unit-Sphere Projection
        # Contrastive objectives behave more predictably when embeddings are
        # normalized before similarity computation. This turns the dot product
        # into cosine similarity and stabilizes the temperature-scaled logits.
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)

        # All-Pairs Similarity Matrix
        # Each anchor is compared against every positive candidate in the batch.
        # This produces the dense similarity table used for both positive
        # matching and negative aggregation.
        logits = (anchor @ positive.transpose(0, 1)) / self.temperature
        exp_logits = torch.exp(logits)

        batch_size = logits.size(0)
        device = logits.device

        # Negative Candidate Mask
        # Begin with all cross-sample pairs as valid negatives, then remove:
        # 1. the diagonal self-pairs,
        # 2. any document pairs known to be neighbors in the citation graph.
        negative_mask = torch.ones((batch_size, batch_size), dtype=torch.bool, device=device)
        negative_mask.fill_diagonal_(False)

        doc_ids = batch_doc_ids.detach().cpu().tolist()

        for row_idx, excluded_ids in enumerate(batch_neighborhoods):
            for col_idx, doc_id in enumerate(doc_ids):
                if int(doc_id) in excluded_ids:
                    negative_mask[row_idx, col_idx] = False

        negative_weights = negative_mask.float()

        # Metadata-Aware Negative Softening
        # Some samples may look like negatives inside the batch but still share
        # venue, publisher, year, or other metadata structure with the anchor.
        # Instead of fully removing them, softly downweight their repulsion term.
        if metadata_affinity is not None:
            affinity = metadata_affinity.to(device=device, dtype=torch.float32)
            negative_weights = torch.where(affinity > 0, negative_weights * self.metadata_negative_weight, negative_weights)

        # Positive Score Extraction
        # Default behavior assumes one aligned positive per row on the diagonal.
        # If a custom positive mask is given, aggregate all allowed positives and
        # average them into a single positive score per anchor.
        if positive_mask is None:
            positive_values = exp_logits.diagonal()
        else:
            pos_weights = positive_mask.to(device=device, dtype=torch.float32)
            positive_values = (exp_logits * pos_weights).sum(dim=1)/ pos_weights.sum(dim=1).clamp_min(1.0)

        # Weighted InfoNCE Denominator
        # The denominator includes the retained negative mass only, after graph
        # masking and metadata-aware downweighting. This reduces false-negative
        # pressure for semantically related scholarly documents.
        negative_values = (exp_logits * negative_weights).sum(dim=1)

        loss = -torch.log(positive_values / (positive_values + negative_values + self.eps))
        return loss.mean()
