from __future__ import annotations

import json
from itertools import cycle
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

from eval import (
    evaluate_predictions, plot_training_history,
    save_evaluation_bundle, save_frame, to_frame)

"""Training orchestration for MetaGraphSci.

This module contains the experiment logic that sits on top of the model:
- self-supervised contrastive pretraining,
- supervised + pseudo-labeled fine-tuning,
- checkpointing and evaluation artifact export.

The implementation deliberately keeps stage-specific logic in the trainer rather
than inside the model. That separation makes the architecture reusable while the
training policy remains easy to swap or document.
"""


class NeighborhoodAwareContrastiveLoss(nn.Module):
    """MCNA contrastive loss with neighborhood masking and metadata-aware downweighting.
    
    This loss function encourages embeddings of positive pairs to be close while pushing 
    apart negative pairs. It modifies standard contrastive loss by ignoring known graph 
    neighbors (preventing false negatives) and applying a softer penalty to documents 
    that share similar metadata (like venue or publisher).
    """

    def __init__(self, temperature: float = 0.07, metadata_negative_weight: float = 0.25) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.metadata_negative_weight = float(metadata_negative_weight)

    def forward(
        self,
        anchor: torch.Tensor, positive: torch.Tensor,
        batch_doc_ids: torch.Tensor, batchneighborhoods: list[set[int]],
        metadata_affinity: torch.Tensor | None = None, positive_mask: torch.Tensor | None = None) -> torch.Tensor:
        
        # Map embeddings to a unit hypersphere to compute cosine similarity via dot product
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)

        # Calculate temperature-scaled similarities between all anchors and positive candidates
        logits = anchor @ positive.t() / self.temperature
        exp_logits = torch.exp(logits)

        batch_size = logits.size(0)
        device = logits.device
        
        # Initialize an all-to-all negative mask, excluding the diagonal (self-comparisons)
        negative_mask = torch.ones((batch_size, batch_size), dtype=torch.bool, device=device)
        negative_mask.fill_diagonal_(False)

        # Exclude known graph neighbors from acting as negatives to avoid penalizing true similarities
        doc_ids = batch_doc_ids.detach().cpu().tolist()
        for row_index, excluded_ids in enumerate(batchneighborhoods):
            for col_index, doc_id in enumerate(doc_ids):
                if int(doc_id) in excluded_ids:
                    negative_mask[row_index, col_index] = False

        negative_weights = negative_mask.float()
        
        # Apply a softer penalty to negatives that share similar metadata (e.g., same venue)
        if metadata_affinity is not None:
            metadata_affinity = metadata_affinity.to(device).float()
            negative_weights = torch.where(
                metadata_affinity > 0,
                negative_weights * self.metadata_negative_weight,
                negative_weights)

        # Identify positive pairs. If no explicit mask is provided, assume the diagonal holds the positives.
        if positive_mask is None:
            positive_values = exp_logits.diag()
        else:
            # Aggregate positive scores based on the provided boolean mask
            positive_weights = positive_mask.to(device).float()
            positive_values = (exp_logits * positive_weights).sum(dim=1) / positive_weights.sum(dim=1).clamp_min(1.0)

        # Sum the weighted negative scores and compute the final InfoNCE-style log loss
        negatives = (exp_logits * negative_weights).sum(dim=1)
        return (-torch.log(positive_values / (positive_values + negatives + 1e-12))).mean()


class PseudoLabeler:
    """Select pseudo-labels with alignment, sharpening, and adaptive thresholds.

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
        self.ema_class_max: torch.Tensor | None = None

    def align(self, probs: torch.Tensor) -> torch.Tensor:
        """Adjusts prediction probabilities to match a target class distribution prior."""
        if not self.distributionalignment:
            return probs

        # Fallback to a uniform prior if no specific target prior is provided
        if self.target_prior is None:
            target_prior = torch.full((probs.size(1),), 1.0 / probs.size(1), device=probs.device)
        else:
            target_prior = self.target_prior.to(probs.device)

        # Normalize predictions against the current batch prior and target prior
        batch_prior = probs.mean(dim=0).clamp_min(1e-8)
        aligned = probs * (target_prior / batch_prior)
        return aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def sharpen(self, probs: torch.Tensor) -> torch.Tensor:
        """Applies temperature scaling to make the probability distribution peakier."""
        if np.isclose(self.temperature, 1.0):
            return probs
        scaled = probs.pow(1.0 / max(self.temperature, 1e-6))
        return scaled / scaled.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def thresholds(self, probs: torch.Tensor) -> torch.Tensor:
        """Maintains a moving average of maximum confidences to establish dynamic thresholds."""
        batch_max = probs.max(dim=0).values.detach()
        if self.ema_class_max is None:
            self.ema_class_max = batch_max
        else:
            self.ema_class_max = self.ema_momentum * self.ema_class_max.to(batch_max.device) + (1.0 - self.ema_momentum) * batch_max
        return self.beta * (self.ema_class_max if self.ema_class_max is not None else batch_max).to(probs.device)

    def select(self, probs: torch.Tensor, epoch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs the full pseudo-labeling pipeline to determine which predictions to keep."""
        adjusted = self.sharpen(self.align(probs))
        confidence, pseudo_labels = adjusted.max(dim=1)
        thresholds = self.thresholds(adjusted)

        # Accept predictions that meet or exceed the dynamic class-specific threshold
        keep = confidence >= thresholds[pseudo_labels]
        
        # Reject all pseudo-labels during the initial warmup phase
        if epoch <= self.warmup_epochs:
            keep = torch.zeros_like(keep)

        # Guarantee a minimum number of accepted pseudo-labels per class to prevent starvation
        if self.min_per_class > 0:
            for cls in range(adjusted.size(1)):
                candidates = torch.where(pseudo_labels == cls)[0]
                if len(candidates) == 0:
                    continue
                if int(keep[candidates].sum().item()) >= self.min_per_class:
                    continue
                topk = confidence[candidates].topk(k=min(self.min_per_class, len(candidates))).indices
                keep[candidates[topk]] = True

        return keep, pseudo_labels, thresholds, adjusted


class MetaGraphSciTrainer:
    """Stages the training and evaluation of MetaGraphSci with a clear separation of concerns.
    
    The trainer orchestrates how and when different training strategies are applied:
    - Stage 1: Self-supervised contrastive pretraining using neighborhood and metadata cues.
    - Stage 2: Semi-supervised fine-tuning balancing standard cross-entropy and pseudo-labels.
    """

    DEFAULTS = {
        "output_dir": "runs/MetaGraphSci", "mixed_precision": "no", "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0, "pretrain_lr": 1e-4, "finetune_lr": 5e-5, "weight_decay": 0.01,
        "selection_metric": "macro_f1","use_mlflow": False, "mlflow_experiment": "MetaGraphSci","run_name": "baseline",
        "contrastive_temperature": 0.07, "metadata_negative_weight": 0.25, "ssl_text_dropout": 0.15, "lambda_ssl": 1.0, "ablation_mode": "full",
        "pseudo_label": {
            "beta": 0.95, "warmup_epochs": 1, "min_per_class": 0,
            "temperature": 1.0, "ema_momentum": 0.9, "distributionalignment": True
        }
    }

    def __init__(
        self, model: nn.Module, citation_graph: Any,
        neighbor_cache: Mapping[int, set[int]] | None = None, config: Mapping[str, object] | None = None,
        label_names: Sequence[str] | None = None, labeled_class_prior: Sequence[float] | None = None) -> None:
        
        # Merge provided config with defaults
        self.cfg = {**self.DEFAULTS, **dict(config or {})}
        pseudo_cfg = {**self.DEFAULTS["pseudo_label"], **dict(self.cfg.get("pseudo_label", {}))}
        self.cfg["pseudo_label"] = pseudo_cfg

        self.model = model
        self.graph = citation_graph
        self.neighbor_cache = {int(k): set(map(int, v)) for k, v in (neighbor_cache or {}).items()}
        self.label_names = list(label_names) if label_names else None

        # Setup filesystem tracking
        self.output_dir = Path(str(self.cfg["output_dir"]))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint = self.output_dir / "checkpoints" / "best_model.pt"
        self.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)

        # Setup distributed and mixed-precision training environments via Accelerate
        self.accelerator = Accelerator(
            gradient_accumulation_steps=int(self.cfg["gradient_accumulation_steps"]),
            mixed_precision=str(self.cfg["mixed_precision"]))
        self.device = self.accelerator.device
        self.model = self.model.to(self.device)

        # Initialize core objectives and utilities
        self.contrastive_loss = NeighborhoodAwareContrastiveLoss(float(self.cfg["contrastive_temperature"]), float(self.cfg["metadata_negative_weight"]))
        self.supervised_loss = nn.CrossEntropyLoss()
        self.pseudo_labeler = PseudoLabeler(target_prior=labeled_class_prior, **pseudo_cfg)

        self.history: list[dict[str, float | int | str]] = []
        self.best_score = float("-inf")
        self.ablation_mode = str(self.cfg["ablation_mode"])

    def optimizer(self, stage: str) -> AdamW:
        """Provisions an optimizer with learning rates tailored to the training stage."""
        learning_rate = float(self.cfg["pretrain_lr"] if stage == "pretrain" else self.cfg["finetune_lr"])
        return AdamW(self.model.parameters(), lr=learning_rate, weight_decay=float(self.cfg["weight_decay"]))

    def extract_context_tensors(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Filters the batch to strictly isolate bounded citation-context tensors."""
        return {k: v for k, v in batch.items() if k.startswith("context_") and isinstance(v, torch.Tensor)}

    def forward(self, batch: Mapping[str, torch.Tensor]):
        """Passes metadata, textual inputs, and citation context through the model architecture."""
        return self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            venue_ids=batch["venue_ids"], publisher_ids=batch["publisher_ids"], author_ids=batch["author_ids"],
            years=batch["years"], ablation_mode=self.ablation_mode, **self.extract_context_tensors(batch))

    def embeddings(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Retrieves raw representation embeddings bypassing classification heads."""
        return self.model.get_embeddings(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            venue_ids=batch["venue_ids"], publisher_ids=batch["publisher_ids"], author_ids=batch["author_ids"],
            years=batch["years"], ablation_mode=self.ablation_mode, **self.extract_context_tensors(batch))

    def augment_batch_for_ssl(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Corrupts the input sequence for self-supervised contrastive learning.
        
        Applies a text-dropout strategy by replacing non-special tokens with mask 
        tokens, forcing the model to reconstruct robust contextual representations.
        """
        augmented = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        input_ids = augmented["input_ids"]
        attention_mask = augmented["attention_mask"].bool()
        probability = float(self.cfg.get("ssl_text_dropout", 0.15))
        
        if probability <= 0.0:
            return augmented
            
        dropout_mask = (torch.rand_like(input_ids.float()) < probability) & attention_mask
        special = (input_ids == 0) | (input_ids == 101) | (input_ids == 102)
        dropout_mask = dropout_mask & ~special
        
        augmented["input_ids"] = input_ids.masked_fill(dropout_mask, 103)
        return augmented

    def metadata_affinity(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Identifies pairs of documents in the batch that share common publication characteristics."""
        venue = batch["venue_ids"]
        publisher = batch["publisher_ids"]
        years = batch["years"].view(-1)
        
        same_venue = venue.unsqueeze(1) == venue.unsqueeze(0)
        same_publisher = publisher.unsqueeze(1) == publisher.unsqueeze(0)
        close_year = (years.unsqueeze(1) - years.unsqueeze(0)).abs() <= (2.0 / 26.0)
        
        affinity = same_venue | same_publisher | close_year
        affinity.fill_diagonal_(False)
        return affinity

    def build_positive_mask(self, batch: Mapping[str, torch.Tensor], metadata_affinity: torch.Tensor) -> torch.Tensor:
        """Determines optimal positive pairings for contrastive learning based on structural topology.
        
        Searches the citation graph cache for neighbors that also exhibit high metadata affinity.
        If no strict topological neighbor is found, falls back to metadata affinity alone.
        """
        doc_ids = batch["doc_id"].detach().cpu().tolist()
        batch_size = len(doc_ids)
        mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=metadata_affinity.device)
        
        for i, doc_id in enumerate(doc_ids):
            neighbors = self.neighbor_cache.get(int(doc_id), set())
            candidates = [j for j, other_id in enumerate(doc_ids) if i != j and int(other_id) in neighbors and bool(metadata_affinity[i, j].item())]
            
            if not candidates:
                candidates = [j for j in range(batch_size) if i != j and bool(metadata_affinity[i, j].item())]
            if candidates:
                mask[i, candidates[0]] = True
        return mask

    def clip(self) -> None:
        """Prevents exploding gradients by capping their maximum norm."""
        self.accelerator.clip_grad_norm_(self.model.parameters(), float(self.cfg["max_grad_norm"]))

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Consolidates tensors across distributed processes."""
        return self.accelerator.gather_for_metrics(tensor).detach().cpu()

    def neighborhoods(self, batch_doc_ids: torch.Tensor) -> list[set[int]]:
        """Retrieves the local citation network connectivity for a batch of documents."""
        return [self.neighbor_cache.get(int(doc_id), set()) for doc_id in batch_doc_ids.detach().cpu().tolist()]

    def save_checkpoint(self, **extra: Any) -> None:
        """Serializes the unwrapped model state directly to disk."""
        payload = {"model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(), **extra}
        torch.save(payload, self.best_checkpoint)

    def load_checkpoint(self) -> None:
        """Rehydrates the unwrapped model state from disk if a checkpoint exists."""
        if self.best_checkpoint.exists():
            state = torch.load(self.best_checkpoint, map_location="cpu")
            self.accelerator.unwrap_model(self.model).load_state_dict(state["model_state_dict"])

    def log_metrics(self, metrics: Mapping[str, float], step: int, prefix: str) -> None:
        """Pushes sanitised scalar metrics to the active MLFlow tracking server."""
        if not bool(self.cfg["use_mlflow"]):
            return
        clean_metrics = {f"{prefix}/{k}": float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)}
        if clean_metrics:
            mlflow.log_metrics(clean_metrics, step=step)

    def start_run(self):
        """Initializes a new MLFlow experiment run and registers the active configuration."""
        if not bool(self.cfg["use_mlflow"]):
            return None
        mlflow.set_experiment(str(self.cfg["mlflow_experiment"]))
        run = mlflow.start_run(run_name=str(self.cfg["run_name"]))
        mlflow.log_params({k: v for k, v in self.cfg.items() if isinstance(v, (int, float, str, bool))})
        return run

    def pretrain(self, loader: DataLoader, optimizer: AdamW, epochs: int, log_interval: int = 20) -> dict[str, list[float]]:
        """Executes the self-supervised contrastive pretraining phase.
        
        Drives the model to learn topological and textual representations by applying 
        augmentations and contrasting corrupted views against uncorrupted representations,
        while respecting local neighborhood structures.
        """
        history = {"epoch": [], "train_loss": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss, total_steps = 0.0, 0.0

            for step, batch in enumerate(loader, start=1):
                with self.accelerator.accumulate(self.model):
                    optimizer.zero_grad()
                    
                    # Generate representations for clean and augmented views
                    anchor = self.embeddings(batch)
                    positive_batch = self.augment_batch_for_ssl(batch)
                    positive = self.embeddings(positive_batch)
                    
                    # Resolve domain-specific positive/negative constraints
                    metadata_affinity = self.metadata_affinity(batch)
                    positive_mask = self.build_positive_mask(batch, metadata_affinity)
                    
                    # NCMA (neighborhood conrastive loss)
                    loss = self.contrastive_loss(anchor, positive, batch["doc_id"], self.neighborhoods(batch["doc_id"]), metadata_affinity=metadata_affinity, positive_mask=positive_mask)
                    
                    self.accelerator.backward(loss)
                    self.clip()
                    optimizer.step()

                total_loss += float(loss.detach().item())
                total_steps += 1
                if step % log_interval == 0 and self.accelerator.is_local_main_process:
                    print(f"[Pretrain] epoch={epoch} step={step}/{len(loader)} loss={loss.item():.4f}")

            average_loss = total_loss / max(total_steps, 1)
            row = {"stage": "pretrain", "epoch": epoch, "train_loss": average_loss}
            history["epoch"].append(epoch)
            history["train_loss"].append(average_loss)
            self.history.append(row)
            self.log_metrics({"train_loss": average_loss}, epoch, "pretrain")

        return history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "val", returnembeddings: bool = False) -> dict[str, Any]:
        """Performs inference on a labeled dataset split to compute standard classification metrics."""
        self.model.eval()

        doc_ids: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        preds: list[torch.Tensor] = []
        probs: list[torch.Tensor] = []
        embeddings: list[torch.Tensor] = []
        total_loss, total_steps = 0, 0

        for batch in loader:
            z, logits, batch_probs = self.forward(batch)
            loss = self.supervised_loss(logits, batch["labels"])
            total_loss += float(loss.detach().item())
            total_steps += 1

            doc_ids.append(self.gather(batch["doc_id"]))
            labels.append(self.gather(batch["labels"]))
            preds.append(self.gather(logits.argmax(dim=1)))
            probs.append(self.gather(batch_probs))
            if returnembeddings:
                embeddings.append(self.gather(z))

        # Collapse distributed tensors into local numpy arrays for evaluation utilities
        y_true = torch.cat(labels).numpy()
        y_pred = torch.cat(preds).numpy()
        y_prob = torch.cat(probs).numpy()
        doc_id_arr = torch.cat(doc_ids).numpy()

        bundle = evaluate_predictions(y_true=y_true, y_pred=y_pred, y_prob=y_prob, doc_ids=doc_id_arr, label_names=self.label_names)
        result: dict[str, Any] = {
            "split": split, "metrics": {"loss": total_loss / max(total_steps, 1), **bundle["metrics"]},
            "bundle": bundle, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "doc_ids": doc_id_arr}
            
        if returnembeddings:
            result["embeddings"] = torch.cat(embeddings).numpy() if embeddings else None

        self.model.train()
        return result

    def finetune(
        self, labeled_loader: DataLoader, unlabeled_loader: DataLoader,
        optimizer: AdamW, epochs: int, val_loader: DataLoader | None = None, log_interval: int = 20) -> dict[str, object]:
        """Executes the semi-supervised fine-tuning phase.
        
        Combines standard cross-entropy loss on ground-truth labeled examples with 
        dynamically generated pseudo-labels computed on a parallel stream of unlabeled data.
        """
        history = {"epoch": [],  "train_loss": [], "sup_loss": [], "ssl_loss": [], "pseudo_label_ratio": []}
        unlabeled_stream = cycle(unlabeled_loader)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss, total_sup, total_ssl = 0.0, 0.0, 0.0
            total_selected, total_unlabeled = 0, 0
            threshold_track: list[float] = []

            for step, labeled_batch in enumerate(labeled_loader, start=1):
                unlabeled_batch = next(unlabeled_stream)
                
                with self.accelerator.accumulate(self.model):
                    optimizer.zero_grad()

                    # Supervised branch
                    _, labeled_logits, _ = self.forward(labeled_batch)
                    sup_loss = self.supervised_loss(labeled_logits, labeled_batch["labels"])

                    # Unlabeled branch: forward pass, selection, and consistency loss
                    _, unlabeled_logits, unlabeled_probs = self.forward(unlabeled_batch)
                    selected, pseudo_labels, thresholds, _ = self.pseudo_labeler.select(unlabeled_probs, epoch=epoch)
                    
                    ssl_loss = self.supervised_loss(unlabeled_logits[selected], pseudo_labels[selected]) if selected.any() else unlabeled_logits.new_zeros(())
                    
                    # Combine objectives
                    loss = sup_loss + float(self.cfg["lambda_ssl"]) * ssl_loss

                    self.accelerator.backward(loss)
                    self.clip()
                    optimizer.step()

                # Accumulate batch statistics
                total_loss += float(loss.detach().item())
                total_sup += float(sup_loss.detach().item())
                total_ssl += float(ssl_loss.detach().item())
                total_selected += int(selected.sum().item())
                total_unlabeled += int(unlabeled_probs.size(0))
                threshold_track.append(float(thresholds.mean().item()))

                if step % log_interval == 0 and self.accelerator.is_local_main_process:
                    print(f"[Finetune] epoch={epoch} step={step}/{len(labeled_loader)} total={loss.item():.4f} sup={sup_loss.item():.4f} ssl={ssl_loss.item():.4f}")

            row: dict[str, float | int | str] = {
                "stage": "finetune", "epoch": epoch,
                "train_loss": total_loss / max(len(labeled_loader), 1),
                "sup_loss": total_sup / max(len(labeled_loader), 1),
                "ssl_loss": total_ssl / max(len(labeled_loader), 1),
                "pseudo_label_ratio": total_selected / max(total_unlabeled, 1),
                "pseudo_threshold_mean": float(np.mean(threshold_track)) if threshold_track else np.nan}

            # Optional validation checkpointing mechanism
            if val_loader is not None:
                val_result = self.evaluate(val_loader, split="val")
                row.update({f"val_{k}": v for k, v in val_result["metrics"].items()})
                
                score = float(val_result["metrics"].get(str(self.cfg["selection_metric"]), val_result["metrics"].get("macro_f1", 0.0)))
                if score > self.best_score:
                    self.best_score = score
                    self.save_checkpoint(epoch=epoch, best_score=score)
                    save_evaluation_bundle(
                        val_result["bundle"], self.output_dir / "artifacts" / "val",
                        "val", val_result["y_true"], val_result["y_pred"], y_prob=val_result["y_prob"],
                        label_names=self.label_names, history_rows=self.history + [row])

            history["epoch"].append(epoch)
            for k in ["train_loss", "sup_loss", "ssl_loss", "pseudo_label_ratio"]:
                history[k].append(row[k])
            self.history.append(row)
            self.log_metrics({k: v for k, v in row.items() if isinstance(v, (float, int))}, epoch, "finetune")

        # Export visualizations and history trails to filesystem
        save_frame(to_frame(self.history), self.output_dir / "artifacts" / "history.csv")
        plot_training_history(self.history, self.output_dir / "artifacts" / "training_curves.png")
        return {"history": history, "history_rows": self.history, "best_score": self.best_score, "best_checkpoint": str(self.best_checkpoint)}

    def train_full_pipeline(
        self, pretrain_loader: DataLoader, labeled_loader: DataLoader, unlabeled_loader: DataLoader,
        val_loader: DataLoader | None = None, test_loader: DataLoader | None = None,
        pretrain_epochs: int = 20, finetune_epochs: int = 10) -> dict[str, object]:
        """Orchestrates the entire end-to-end training lifecycle.
        
        Registers all components with the hardware accelerator, executes the pretraining 
        run, transitions into semi-supervised fine-tuning, and wraps up by loading 
        the best performing state to evaluate on the hold-out test set.
        """
        pretrainoptimizer = self.optimizer("pretrain")
        finetuneoptimizer = self.optimizer("finetune")

        # Compile component lists dynamically to accommodate missing validation/test loaders
        items: list[Any] = [self.model, pretrainoptimizer, finetuneoptimizer, pretrain_loader, labeled_loader, unlabeled_loader]
        if val_loader is not None:
            items.append(val_loader)
        if test_loader is not None:
            items.append(test_loader)

        # Apply multi-device and mixed-precision wrapping
        prepared = self.accelerator.prepare(*items)
        self.model = prepared[0]
        pretrainoptimizer = prepared[1]
        finetuneoptimizer = prepared[2]
        pretrain_loader = prepared[3]
        labeled_loader = prepared[4]
        unlabeled_loader = prepared[5]

        cursor = 6  
        if val_loader is not None:
            val_loader = prepared[cursor]
            cursor += 1
        if test_loader is not None:
            test_loader = prepared[cursor]

        run = self.start_run()
        try:
            # Sequential execution of training stages
            pretrain_result = self.pretrain(pretrain_loader, pretrainoptimizer, epochs=pretrain_epochs)
            finetune_result = self.finetune(labeled_loader, unlabeled_loader, finetuneoptimizer, epochs=finetune_epochs, val_loader=val_loader)
            
            # Post-training evaluation on best weights
            test_result = None
            if test_loader is not None:
                self.load_checkpoint()
                test_result = self.evaluate(test_loader, split="test", returnembeddings=True)
                artifact_dir = self.output_dir / "artifacts" / "test"
                save_evaluation_bundle(
                    test_result["bundle"], artifact_dir, "test",
                    test_result["y_true"], test_result["y_pred"], y_prob=test_result["y_prob"],
                    embeddings=test_result.get("embeddings"), label_names=self.label_names, history_rows=self.history)
                (artifact_dir / "test_metrics.json").write_text(json.dumps(test_result["metrics"], indent=2))

            if bool(self.cfg["use_mlflow"]):
                mlflow.log_artifacts(str(self.output_dir / "artifacts"))

            return {"pretrain": pretrain_result, "finetune": finetune_result, "test": test_result, "history_rows": self.history}
        finally:
            if run is not None:
                mlflow.end_run()
