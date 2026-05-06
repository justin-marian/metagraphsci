from typing import Any, Sequence
import numpy as np
import torch
from torch import Tensor


class PseudoLabeler:
    """
    Generates pseudo-targets through prior correction, confidence sharpening,
    and adaptive acceptance criteria.

    PERSISTENT ADAPTIVE STATE
    -------------------------
    ema_class_max must be saved and restored explicitly alongside the model
    checkpoint.  Losing it silently resets the curriculum to cold-start.
    """

    def __init__(
        self, beta: float = 0.80, warmup_epochs: int = 3, min_per_class: int = 0,
        temperature: float = 0.5, ema_momentum: float = 0.95,
        distributionalignment: bool = True,
        target_prior: Sequence[float] | None = None) -> None:
        # HYPERPARAMETER CHANGES:
        #   beta: 0.95 → 0.80  (less restrictive threshold — accepts more pseudo-labels
        #                        early without waiting for near-perfect confidence)
        #   warmup_epochs: 1 → 3  (avoids pseudo-labeling from an unstable model)
        #   temperature: 1.0 → 0.5  (sharper predictions are more reliable targets)
        #   ema_momentum: 0.90 → 0.95  (smoother curriculum evolution)
        self.beta               = float(beta)
        self.warmup_epochs      = int(warmup_epochs)
        self.min_per_class      = int(min_per_class)
        self.temperature        = float(temperature)
        self.ema_momentum       = float(ema_momentum)
        self.distributionalignment = bool(distributionalignment)
        self.target_prior       = (None if target_prior is None
            else torch.tensor(target_prior, dtype=torch.float32))
        self.ema_class_max: Tensor | None = None

    def labeler_state_dict(self) -> dict[str, Any]:
        return {
            "ema_class_max": self.ema_class_max.cpu() if self.ema_class_max is not None else None,
            "target_prior":  self.target_prior.cpu()  if self.target_prior  is not None else None,
        }

    def load_labeler_state_dict(self, state: dict[str, Any]) -> None:
        """Restores adaptive state from a checkpoint.

        BUG FIXED: the original guard was `not isinstance(ema, Tensor)`, so the
        block was entered only when ema was NOT a Tensor.  The inner expression
        `ema.clone() if isinstance(ema, Tensor) else None` was then always False,
        silently resetting ema_class_max to None on every checkpoint load and
        collapsing the adaptive curriculum back to cold-start.
        """
        ema = state.get("ema_class_max")
        if isinstance(ema, Tensor):
            self.ema_class_max = ema.clone()
        else:
            # Warn when loading a checkpoint written by the old (broken)
            # code that always saved ema_class_max=None.  Silently resetting to
            # cold-start caused the adaptive curriculum to degrade without notice.
            if "ema_class_max" in state:
                import warnings
                warnings.warn(
                    "PseudoLabeler checkpoint contains ema_class_max=None "
                    "Adaptive curriculum will restart from cold-start.",
                    UserWarning, stacklevel=2)
        prior = state.get("target_prior")
        if isinstance(prior, Tensor):
            self.target_prior = prior.clone()

    def reset(self) -> None:
        self.ema_class_max = None

    def align(self, probs: Tensor) -> Tensor:
        """Calibrates predictions toward the expected class distribution."""
        if not self.distributionalignment:
            return probs
        if self.target_prior is None:
            target_prior = torch.full((probs.size(1),), 1.0 / probs.size(1), device=probs.device)
        else:
            target_prior = self.target_prior.to(probs.device)
        batch_prior = probs.mean(dim=0).clamp_min(1e-9)
        aligned = probs * (target_prior / batch_prior)
        return aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-9)

    def sharpen(self, probs: Tensor) -> Tensor:
        """Reduces uncertainty by sharpening the probability distribution."""
        if np.isclose(self.temperature, 1.0):
            return probs
        scaled = probs.pow(1.0 / max(self.temperature, 1e-6))
        return scaled / scaled.sum(dim=1, keepdim=True).clamp_min(1e-9)

    def thresholds(self, probs: Tensor) -> Tensor:
        """Updates per-class confidence requirements via EMA of observed maxima."""
        batch_max = probs.max(dim=0).values.detach()
        if self.ema_class_max is None:
            self.ema_class_max = batch_max
        else:
            self.ema_class_max = (
                self.ema_momentum * self.ema_class_max.to(batch_max.device)
                + (1.0 - self.ema_momentum) * batch_max)
        return self.beta * self.ema_class_max.to(probs.device)

    def select(self, probs: Tensor, epoch: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Combines calibration, sharpening, adaptive filtering, and min-per-class safety."""
        adjusted = self.sharpen(self.align(probs))
        confidence, pseudo_labels = adjusted.max(dim=1)
        thresholds = self.thresholds(adjusted)

        keep = confidence >= thresholds[pseudo_labels]

        if epoch <= self.warmup_epochs:
            # Warmup must be absolute. Do not allow min_per_class to re-enable
            # pseudo-labels while the classifier is still near-random.
            return torch.zeros_like(keep), pseudo_labels, thresholds, adjusted

        if self.min_per_class > 0:
            for cls in range(adjusted.size(1)):
                candidates = torch.where(pseudo_labels == cls)[0]
                if len(candidates) == 0 or int(keep[candidates].sum().item()) >= self.min_per_class:
                    continue
                topk = confidence[candidates].topk(
                    k=min(self.min_per_class, len(candidates))).indices
                keep[candidates[topk]] = True

        return keep, pseudo_labels, thresholds, adjusted
