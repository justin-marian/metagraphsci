import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class-weight (alpha) and label smoothing.

    L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self, gamma: float = 2.0, alpha: Tensor | None = None,
        label_smoothing: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        num_classes = logits.size(-1)

        if self.label_smoothing > 0.0:
            with torch.no_grad():
                true_dist = torch.full_like(log_probs, self.label_smoothing / max(num_classes - 1, 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(true_dist * log_probs).sum(dim=-1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
        else:
            ce = F.nll_loss(log_probs, targets, reduction="none")
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)

        focal_factor = (1.0 - pt) ** self.gamma
        loss = focal_factor * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


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
        anchor: Tensor, positive: Tensor,
        batch_doc_ids: Tensor, batch_neighborhoods: list[set[int]],
        metadata_affinity: Tensor | None = None, positive_mask: Tensor | None = None
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

        # Numerical stability: subtract per-row max before exp. The shift cancels
        # between numerator and denominator of the InfoNCE ratio, so the loss is
        # mathematically unchanged but exp() stays bounded under bf16/fp16 autocast.
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits)

        batch_size = logits.size(0)
        device = logits.device

        # Negative Candidate Mask
        # Begin with all cross-sample pairs as valid negatives, then remove:
        # 1. the diagonal self-pairs,
        # 2. any document pairs known to be neighbors in the citation graph.
        negative_mask = torch.ones((batch_size, batch_size), dtype=torch.bool, device=device)
        negative_mask.fill_diagonal_(False)

        # Convert neighborhoods into a padded tensor for vectorized comparison
        max_neighbors = max(len(n) for n in batch_neighborhoods) if batch_neighborhoods else 0

        if max_neighbors > 0:
            neighbors_tensor = torch.full((batch_size, max_neighbors), fill_value=-1, dtype=batch_doc_ids.dtype, device=device)
            for i, neigh in enumerate(batch_neighborhoods):
                if len(neigh) > 0:
                    neighbors_tensor[i, :len(neigh)] = torch.tensor(list(neigh), device=device, dtype=batch_doc_ids.dtype)

            # Compare (B,1) vs (B,K) - (B,B,K)
            match = batch_doc_ids.unsqueeze(0).unsqueeze(-1) == neighbors_tensor.unsqueeze(1)
            # Collapse neighbor dimension - (B,B)
            is_neighbor = match.any(dim=-1)

            negative_mask = negative_mask & (~is_neighbor)

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
            # Anchor and positive are augmentations of the same doc, so the diagonal
            # is always a valid SimCLR-style positive. Fall back to it for any row
            # whose topology/metadata mask is empty — otherwise (exp * 0).sum() = 0
            # propagates as -log(0) = +inf and poisons the gradient with NaN.
            pos_mask = positive_mask.to(device=device, dtype=torch.bool).clone()
            empty_rows = ~pos_mask.any(dim=1)
            if empty_rows.any():
                pos_mask[empty_rows, torch.arange(batch_size, device=device)[empty_rows]] = True
            pos_weights = pos_mask.to(dtype=torch.float32)
            positive_values = ((exp_logits * pos_weights).sum(dim=1) / pos_weights.sum(dim=1).clamp_min(1.0))

        # Weighted InfoNCE Denominator
        # The denominator includes the retained negative mass only, after graph
        # masking and metadata-aware downweighting. This reduces false-negative
        # pressure for semantically related scholarly documents.
        negative_values = (exp_logits * negative_weights).sum(dim=1)

        loss = -torch.log(positive_values / (positive_values + negative_values + self.eps))
        return loss.mean()
