from typing import Any, Sequence
import numpy as np
import torch
from torch import Tensor


class PseudoLabeler:
    """
    Generates pseudo-targets through prior correction, confidence sharpening,
    and adaptive acceptance criteria.

    Intended for semi-supervised learning, where predictions on unlabeled samples 
    are filtered before being reused as supervision. Its role is to reduce confirmation bias, 
    strengthen confident decisions, and apply a dynamic curriculum so that easier categories
    do not dominate training too early.

    PERSISTENT ADAPTIVE STATE
    -------------------------
    Part of the internal adaptive behavior depends on a running estimate that evolves
    over time. Because this state is not automatically preserved by the main training
    checkpoint mechanism, it should be saved and restored explicitly when resuming
    training. Otherwise, the acceptance curriculum restarts from scratch and may
    behave inconsistently after loading a checkpoint.
    """

    def __init__(
        self, beta: float = 0.95, warmup_epochs: int = 0, min_per_class: int = 0,
        temperature: float = 1.0, ema_momentum: float = 0.9, distributionalignment: bool = True,
        target_prior: Sequence[float] | None = None) -> None:
        self.beta = float(beta)
        self.warmup_epochs = int(warmup_epochs)
        self.min_per_class = int(min_per_class)
        self.temperature = float(temperature)
        self.ema_momentum = float(ema_momentum)
        self.distributionalignment = bool(distributionalignment)
        self.target_prior = None if target_prior is None else torch.tensor(target_prior, dtype=torch.float32)
        self.ema_class_max: Tensor | None = None

    def labeler_state_dict(self) -> dict[str, Any]:
        """Exports the adaptive state required for consistent checkpoint restoration."""
        return {
            "ema_class_max": self.ema_class_max.cpu() if self.ema_class_max is not None else None,
            "target_prior": self.target_prior.cpu() if self.target_prior is not None else None
        }

    def load_labeler_state_dict(self, state: dict[str, Any]) -> None:
        """Restores the adaptive state previously saved during training."""
        ema = state.get("ema_class_max")
        if ema is not None and not isinstance(ema, Tensor):
            self.ema_class_max = ema.clone() if isinstance(ema, Tensor) else None
        prior = state.get("target_prior")
        if isinstance(prior, Tensor) and prior is not None:
            self.target_prior = prior.clone()

    def reset(self) -> None:
        """Removes accumulated history before beginning a separate run."""
        self.ema_class_max = None

    def align(self, probs: Tensor) -> Tensor:
        """
        Calibrates predictions so that the resulting category proportions better match
        an expected target distribution.
        """
        # Without correction, pseudo-target generation tends to over-favor dominant
        # categories and suppress underrepresented ones.
        # This stage counteracts that tendency by reweighting predictions toward a
        # desired prior, improving balance throughout semi-supervised training.
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
        """Reduces uncertainty by making confident outcomes more pronounced."""
        # Soft or ambiguous predictions are risky when converted into supervision.
        # This transformation suppresses uncertainty and makes the most likely outcome
        # more dominant, which improves the reliability of accepted pseudo-targets.
        if np.isclose(self.temperature, 1.0):
            return probs

        scaled = probs.pow(1.0 / max(self.temperature, 1e-6))
        return scaled / scaled.sum(dim=1, keepdim=True).clamp_min(1e-9)

    def thresholds(self, probs: Tensor) -> Tensor:
        """Updates confidence requirements dynamically according to the model's evolving behavior."""
        # Fixed acceptance rules are often too strict for difficult categories and
        # too lenient for easy ones.
        # A moving estimate provides a curriculum-like mechanism that adapts over
        # time, allowing participation to grow naturally as reliability improves.
        #
        # Because this behavior depends on accumulated history, that history should
        # be preserved across save/load boundaries to avoid silently resetting the curriculum.
        batch_max = probs.max(dim=0).values.detach()

        if self.ema_class_max is None:
            self.ema_class_max = batch_max
        else:
            self.ema_class_max = self.ema_momentum * self.ema_class_max.to(batch_max.device) + (1.0 - self.ema_momentum) * batch_max

        return self.beta * self.ema_class_max.to(probs.device)

    def select(self, probs: Tensor, epoch: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Combines calibration, sharpening, adaptive filtering, and safety constraints
        to decide which pseudo-targets are retained.
        """
        adjusted = self.sharpen(self.align(probs))
        confidence, pseudo_labels = adjusted.max(dim=1)
        thresholds = self.thresholds(adjusted)

        keep = confidence >= thresholds[pseudo_labels]

        # Early training stages are typically too unstable for reliable self-generated
        # supervision, so acceptance is delayed until a minimum maturity is reached.
        if epoch <= self.warmup_epochs:
            keep = torch.zeros_like(keep)

        # An additional safeguard prevents difficult categories from disappearing
        # entirely during training by ensuring they continue to contribute learning
        # signal even when confidence remains comparatively low.
        if self.min_per_class > 0:
            for cls in range(adjusted.size(1)):
                candidates = torch.where(pseudo_labels == cls)[0]
                if len(candidates) == 0 or int(keep[candidates].sum().item()) >= self.min_per_class:
                    continue

                topk = confidence[candidates].topk(k=min(self.min_per_class, len(candidates))).indices
                keep[candidates[topk]] = True

        return keep, pseudo_labels, thresholds, adjusted
