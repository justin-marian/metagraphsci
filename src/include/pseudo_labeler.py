from typing import Sequence
import numpy as np
import torch
from torch import Tensor


class PseudoLabeler:
    """
    Select pseudo-labels with alignment, sharpening, and adaptive thresholds.

    This utility processes model predictions on unlabeled data to generate high-quality 
    pseudo-labels for semi-supervised training. It corrects for class imbalance, 
    amplifies confidence, and uses dynamic thresholds to accept only robust predictions.
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

    def align(self, probs: Tensor) -> Tensor:
        """Adjusts prediction probabilities to match a target class distribution prior."""
        # Distribution Alignment
        # In semi-supervised learning, models suffer from severe "confirmation bias", they 
        # confidently predict majority classes on unlabeled data and ignore minority classes.
        # By scaling the model's current batch predictions against the known ground-truth 
        # distribution (target_prior), force the model to generate pseudo-labels that respect
        # the true dataset imbalance, preventing minority class collapse.
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
        """Temperature scaling to make the probability distribution peakier."""
        # Entropy Minimization
        # Pseudo-labels are used as hard targets (one-hot vectors). If the model's 
        # probability distribution is "flat" (e.g., [0.4, 0.3, 0.3]), taking the argmax 
        # is dangerous. Temperature scaling (T < 1.0) artificially pushes the highest 
        # probability closer to 1.0 and suppresses the others. This acts as an implicit 
        # entropy minimization regularizer during training.
        if np.isclose(self.temperature, 1.0):
            return probs
            
        scaled = probs.pow(1.0 / max(self.temperature, 1e-6))
        return scaled / scaled.sum(dim=1, keepdim=True).clamp_min(1e-9)

    def thresholds(self, probs: Tensor) -> Tensor:
        """Maintains a moving average of maximum confidences to establish dynamic thresholds."""
        # Dynamic Curriculum (FlexMatch approach)
        # A static threshold (e.g., 0.95) works poorly because "easy" classes cross 0.95 
        # in epoch 1, while "hard" classes might never cross it, causing them to receive 
        # zero pseudo-labels. Track an Exponential Moving Average (EMA) of the model's 
        # maximum confidence *per class*. This creates a dynamic threshold: hard classes 
        # are allowed to have lower thresholds initially, dynamically scaling up as the model learns.
        batch_max = probs.max(dim=0).values.detach()
        
        if self.ema_class_max is None:
            self.ema_class_max = batch_max
        else:
            self.ema_class_max = self.ema_momentum * self.ema_class_max.to(batch_max.device) + (1.0 - self.ema_momentum) * batch_max
            
        return self.beta * (self.ema_class_max if self.ema_class_max is not None else batch_max).to(probs.device)

    def select(self, probs: Tensor, epoch: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Runs the full pseudo-labeling pipeline to determine which predictions to keep."""
        adjusted = self.sharpen(self.align(probs))
        confidence, pseudo_labels = adjusted.max(dim=1)
        thresholds = self.thresholds(adjusted)

        keep = confidence >= thresholds[pseudo_labels]
        
        # Warmup Phase
        # At epoch 0, the model's weights are random (or only pretrained via self-supervision). 
        # Generating pseudo-labels immediately would permanently encode random garbage 
        # into the training loop. Strictly reject everything until the warmup phase clears.
        if epoch <= self.warmup_epochs:
            keep = torch.zeros_like(keep)

        # Anti-Starvation Fallback
        # Even with distribution alignment, a catastrophic batch might result in 0 accepted 
        # labels for a specific class. `min_per_class` forces the model to accept the top-K 
        # most confident predictions for that class, regardless of threshold, ensuring the 
        # cross-entropy loss always receives gradients for every category.
        if self.min_per_class > 0:
            for cls in range(adjusted.size(1)):
                candidates = torch.where(pseudo_labels == cls)[0]
                if len(candidates) == 0 or int(keep[candidates].sum().item()) >= self.min_per_class:
                    continue
                    
                topk = confidence[candidates].topk(k=min(self.min_per_class, len(candidates))).indices
                keep[candidates[topk]] = True

        return keep, pseudo_labels, thresholds, adjusted
