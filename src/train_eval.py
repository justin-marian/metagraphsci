import json
from itertools import cycle
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlflow
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from loguru import logger
from tqdm.auto import tqdm

from .include import (
    NeighborhoodAwareContrastiveLoss, PseudoLabeler,
    evaluate_predictions, plot_training_history,
    save_evaluation_bundle, save_frame,
    setup_global_logger, to_frame)


"""
Training orchestration for MetaGraphSci.

This module contains the experiment logic that sits on top of the model:
- self-supervised contrastive pretraining,
- supervised + pseudo-labeled fine-tuning,
- checkpointing and evaluation artifact export.

The implementation deliberately keeps stage-specific logic in the trainer rather
than inside the model. That separation makes the architecture reusable while the
training policy remains easy to swap or document.
"""


class MetaGraphSciTrainerEval:
    """Stages the training and evaluation of MetaGraphSci with a clear separation of concerns.

    The trainer orchestrates how and when different training strategies are applied:
    - Stage 1: Self-supervised contrastive pretraining using neighborhood and metadata cues.
    - Stage 2: Semi-supervised fine-tuning balancing standard cross-entropy and pseudo-labels.
    """

    DEFAULTS = {
        "output_dir": "../out/", "mixed_precision": "no", "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0, "pretrain_lr": 1e-4, "finetune_lr": 5e-5, "weight_decay": 0.01,
        "selection_metric": "macro_f1", "run_name": "baseline",

        # Dual Tracker Support
        "use_mlflow": False, "mlflow_experiment": "MetaGraphSci",
        "use_wandb": False, "wandb_project": "MetaGraphSci",

        "contrastive_temperature": 0.07, "metadata_negative_weight": 0.25,
        "ssl_text_dropout": 0.15, "lambda_ssl": 1.0, "ablation_mode": "full",
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
        self.label_names = list(label_names) if label_names is not None else None

        # Setup filesystem tracking
        self.output_dir = Path(str(self.cfg["output_dir"]))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint = self.output_dir / "checkpoints" / "best_model.pt"
        self.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)

        setup_global_logger(self.output_dir)
        logger.info(f"Initialized MetaGraphSciTrainerEval. Artifacts will be saved to: {self.output_dir}")

        # Scalable Hardware Abstraction
        # HuggingFace Accelerate automatically handles mixed precision (fp16/bf16),
        # multi-GPU gradient synchronization, and gradient accumulation. This allows
        # the exact same trainer code to run on a single local GPU or a massive SLURM cluster.
        self.accelerator = Accelerator(
            gradient_accumulation_steps=int(self.cfg["gradient_accumulation_steps"]),
            mixed_precision=str(self.cfg["mixed_precision"]))
        self.device = self.accelerator.device
        self.model = self.model.to(self.device)

        # Initialize core objectives and utilities.
        pseudo_cfg_for_init = {k: v for k, v in pseudo_cfg.items() if k != "target_prior"}
        self.contrastive_loss = NeighborhoodAwareContrastiveLoss(
            float(self.cfg["contrastive_temperature"]),
            float(self.cfg["metadata_negative_weight"]))
        self.supervised_loss = nn.CrossEntropyLoss()
        self.pseudo_labeler = PseudoLabeler(target_prior=labeled_class_prior, **pseudo_cfg_for_init)

        self.history: list[dict[str, float | int | str]] = []
        self.best_score = float("-inf")
        self.ablation_mode = str(self.cfg["ablation_mode"])

    def optimizer(self, stage: str) -> AdamW:
        """Provisions an optimizer with learning rates tailored to the training stage."""
        learning_rate = float(self.cfg["pretrain_lr"] if stage == "pretrain" else self.cfg["finetune_lr"])
        return AdamW(self.model.parameters(), lr=learning_rate, weight_decay=float(self.cfg["weight_decay"]))

    def extract_context_tensors(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Filters the batch to strictly isolate bounded citation-context tensors."""
        return {k: v for k, v in batch.items() if k.startswith("context_") and isinstance(v, Tensor)}

    def forward(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Passes metadata, textual inputs, and citation context through the model architecture."""
        return self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            venue_ids=batch["venue_ids"], publisher_ids=batch["publisher_ids"], author_ids=batch["author_ids"],
            years=batch["years"], ablation_mode=self.ablation_mode, **self.extract_context_tensors(batch))

    def embeddings(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Retrieves raw representation embeddings bypassing classification heads."""
        return self.model.get_embeddings(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            venue_ids=batch["venue_ids"], publisher_ids=batch["publisher_ids"], author_ids=batch["author_ids"],
            years=batch["years"], ablation_mode=self.ablation_mode, **self.extract_context_tensors(batch))

    def augment_batch_for_ssl(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Corrupts the input sequence for self-supervised contrastive learning."""
        # By randomly replacing text tokens with [MASK] (ID 103), simulate missing
        # or degraded abstracts. This forces the cross-modal fusion to rely heavily on
        # the citation graph and metadata streams, preventing the text encoder from
        # becoming a "lazy shortcut" during contrastive pretraining.
        augmented = {key: value.clone() if isinstance(value, Tensor) else value for key, value in batch.items()}
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

    def metadata_affinity(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Identifies pairs of documents in the batch that share common publication characteristics."""
        # Semantic Topology Filtering
        # Citations are notoriously noisy (e.g., self-citations, obligatory background cites).
        # Construct a boolean mask representing "Metadata Affinity" (shared venue, shared
        # publisher, or adjacent publication year). This acts as a semantic filter for the graph.
        #
        # NOTE: years are expected to be normalised to [0, 1] by the dataset loader.
        # The threshold 2.0/26.0 corresponds to a 2-year window over a 26-year span.
        # If raw integer years are passed instead, this comparison will silently produce
        # all-False results. Verify the dataset year normalisation before adjusting.
        venue = batch["venue_ids"]
        publisher = batch["publisher_ids"]
        years = batch["years"].view(-1)

        same_venue = venue.unsqueeze(1) == venue.unsqueeze(0)
        same_publisher = publisher.unsqueeze(1) == publisher.unsqueeze(0)
        close_year = (years.unsqueeze(1) - years.unsqueeze(0)).abs() <= (2.0 / 26.0)

        affinity = same_venue | same_publisher | close_year
        affinity.fill_diagonal_(False)
        return affinity

    def build_positive_mask(self, batch: Mapping[str, Tensor], metadata_affinity: Tensor) -> Tensor:
        """Determines optimal positive pairings for contrastive learning based on structural topology."""
        # Topologically Grounded Positives
        # For standard SimCLR, positives are just augmentations of the same image.
        # For MetaGraphSci, a true positive must be structurally adjacent
        # (in the neighbor cache) AND semantically aligned (high metadata affinity).
        #
        # BUG FIXED: the original code only activated mask[i, candidates[0]], accepting
        # at most one positive per anchor. InfoNCE is designed to handle multiple positives
        # and deliberately aggregating all available topology-grounded positives gives the
        # loss more signal per step. All structurally and semantically aligned pairs within
        # the batch are now marked as positives, with fallback to metadata-only pairs when
        # no neighbour is present in the batch.
        doc_ids = batch["doc_id"].detach().cpu().tolist()
        batch_size = len(doc_ids)
        mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=metadata_affinity.device)

        for i, doc_id in enumerate(doc_ids):
            neighbors = self.neighbor_cache.get(int(doc_id), set())
            # Prefer pairs that are both graph-adjacent and metadata-compatible
            candidates = [
                j for j, other_id in enumerate(doc_ids)
                if i != j and int(other_id) in neighbors and bool(metadata_affinity[i, j].item())
            ]
            # Fall back to metadata-only affinity when no cited neighbour is in the batch
            if not candidates:
                candidates = [j for j in range(batch_size) if i != j and bool(metadata_affinity[i, j].item())]

            # Mark ALL valid positives, not just the first one
            for j in candidates:
                mask[i, j] = True

        return mask

    def clip(self) -> None:
        """Prevents exploding gradients by capping their maximum norm."""
        self.accelerator.clip_grad_norm_(self.model.parameters(), float(self.cfg["max_grad_norm"]))

    def gather(self, tensor: Tensor) -> Tensor:
        """Consolidates tensors across distributed processes."""
        return self.accelerator.gather_for_metrics(tensor).detach().cpu()

    def neighborhoods(self, batch_doc_ids: Tensor) -> list[set[int]]:
        """Retrieves the local citation network connectivity for a batch of documents."""
        return [self.neighbor_cache.get(int(doc_id), set()) for doc_id in batch_doc_ids.detach().cpu().tolist()]

    def save_checkpoint(self, **extra: Any) -> None:
        """Serializes the unwrapped model state and pseudo-labeler EMA state to disk.

        BUG FIXED: the original implementation only saved model_state_dict. After
        reloading the best checkpoint for test evaluation, pseudo_labeler.ema_class_max
        was silently reset to None, collapsing the dynamic curriculum thresholds back
        to their cold-start state. The pseudo-labeler EMA is now persisted alongside
        the model so that load_checkpoint() fully restores the training state.
        """
        payload = {
            "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
            "pseudo_labeler": self.pseudo_labeler.labeler_state_dict(),
            **extra,
        }
        torch.save(payload, self.best_checkpoint)

    def load_checkpoint(self) -> None:
        """Rehydrates the unwrapped model state and pseudo-labeler EMA from disk."""
        if self.best_checkpoint.exists():
            state = torch.load(self.best_checkpoint, map_location="cpu")
            self.accelerator.unwrap_model(self.model).load_state_dict(state["model_state_dict"])
            # Restore pseudo-labeler EMA if present (checkpoints written before this fix
            # won't have the key; silently skip in that case for backward compatibility).
            if "pseudo_labeler" in state:
                self.pseudo_labeler.load_labeler_state_dict(state["pseudo_labeler"])

    def log_metrics(self, metrics: Mapping[str, float], step: int, prefix: str) -> None:
        """Pushes sanitised scalar metrics to MLFlow and/or W&B."""
        clean_metrics = {f"{prefix}/{k}": float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)}
        if not clean_metrics:
            return

        if bool(self.cfg.get("use_mlflow")):
            mlflow.log_metrics(clean_metrics, step=step)
        if bool(self.cfg.get("use_wandb")):
            wandb.log(clean_metrics, step=step)

    def start_run(self) -> None:
        """Initializes tracking for MLFlow and/or W&B based on the config."""
        flat_config = {k: v for k, v in self.cfg.items() if isinstance(v, (int, float, str, bool, dict))}

        if bool(self.cfg.get("use_mlflow")):
            mlflow.set_experiment(str(self.cfg["mlflow_experiment"]))
            mlflow.start_run(run_name=str(self.cfg["run_name"]))
            mlflow.log_params({k: v for k, v in self.cfg.items() if isinstance(v, (int, float, str, bool))})

        if bool(self.cfg.get("use_wandb")):
            wandb.init(project=str(self.cfg["wandb_project"]), name=str(self.cfg["run_name"]), config=flat_config, reinit=True)

    def pretrain(self, loader: DataLoader, optimizer: AdamW, epochs: int, log_interval: int = 20) -> dict[str, list[float]]:
        """Executes the self-supervised contrastive pretraining phase."""
        history = {"epoch": [], "train_loss": []}
        show_bar = self.accelerator.is_local_main_process
        epoch_bar = tqdm(range(1, epochs + 1), desc="Pretrain", unit="ep", disable=not show_bar)

        for epoch in epoch_bar:
            self.model.train()
            total_loss, total_steps = 0.0, 0.0
            step_bar = tqdm(
                loader, total=len(loader), desc=f"Pretrain ep {epoch}/{epochs}",
                unit="batch", leave=False, dynamic_ncols=True, disable=not show_bar)
            for step, batch in enumerate(step_bar, start=1):
                with self.accelerator.accumulate(self.model):
                    optimizer.zero_grad()

                    # Generate representations for clean and augmented views
                    anchor = self.embeddings(batch)
                    positive_batch = self.augment_batch_for_ssl(batch)
                    positive = self.embeddings(positive_batch)

                    metadata_affinity = self.metadata_affinity(batch)
                    positive_mask = self.build_positive_mask(batch, metadata_affinity)

                    # NCMA (neighborhood contrastive loss)
                    loss = self.contrastive_loss(
                        anchor, positive, batch["doc_id"], self.neighborhoods(batch["doc_id"]),
                        metadata_affinity=metadata_affinity, positive_mask=positive_mask)

                    self.accelerator.backward(loss)
                    self.clip()
                    optimizer.step()

                loss_value = float(loss.detach().item())
                total_loss += loss_value
                total_steps += 1
                step_bar.set_postfix(loss=f"{loss_value:.4f}")

                if step % log_interval == 0 and show_bar:
                    tqdm.write(f"[Pretrain] epoch={epoch} step={step}/{len(loader)} loss={loss_value:.4f}")

            step_bar.close()
            average_loss = total_loss / max(total_steps, 1)
            row = {"stage": "pretrain", "epoch": epoch, "train_loss": average_loss}

            if show_bar:
                logger.info(f"[Pretrain] epoch={epoch}/{epochs} avg_loss={average_loss:.4f}")
            epoch_bar.set_postfix(avg_loss=f"{average_loss:.4f}")

            history["epoch"].append(epoch)
            history["train_loss"].append(average_loss)
            self.history.append(row)
            self.log_metrics({"train_loss": average_loss}, epoch, "pretrain")

        epoch_bar.close()
        return history

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader, split: str = "val", return_embeddings: bool = False,
        output_dir: Path | None = None, history_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        """Performs inference and optionally generates all evaluation plots and artifacts."""
        self.model.eval()

        doc_ids: list[Tensor] = []
        labels: list[Tensor] = []
        preds: list[Tensor] = []
        probs: list[Tensor] = []
        embeddings: list[Tensor] = []
        total_loss, total_steps = 0.0, 0.0
        show_bar = self.accelerator.is_local_main_process
        eval_bar = tqdm(
            loader, total=len(loader), desc=f"Eval[{split}]",
            unit="batch", leave=False, dynamic_ncols=True, disable=not show_bar)

        for batch in eval_bar:
            z, logits, batch_probs = self.forward(batch)
            loss = self.supervised_loss(logits, batch["labels"])
            total_loss += float(loss.detach().item())
            total_steps += 1

            doc_ids.append(self.gather(batch["doc_id"]).detach().cpu())
            labels.append(self.gather(batch["labels"]).detach().cpu())
            preds.append(self.gather(logits.argmax(dim=1)).detach().cpu())
            probs.append(self.gather(batch_probs).detach().cpu())

            if return_embeddings or output_dir is not None:
                embeddings.append(self.gather(z).detach().cpu())

            eval_bar.set_postfix(loss=f"{total_loss / max(total_steps, 1):.4f}")

        eval_bar.close()
        # Collapse distributed tensors into local numpy arrays for evaluation utilities
        y_true = torch.cat(labels).numpy()
        y_pred = torch.cat(preds).numpy()
        y_prob = torch.cat(probs).numpy()
        doc_id_arr = torch.cat(doc_ids).numpy()
        emb_arr = torch.cat(embeddings).numpy() if embeddings else None

        bundle = evaluate_predictions(y_true=y_true, y_pred=y_pred, y_prob=y_prob, doc_ids=doc_id_arr, label_names=self.label_names)
        result: dict[str, Any] = {
            "split": split, "metrics": {"loss": total_loss / max(total_steps, 1), **bundle["metrics"]},
            "bundle": bundle, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "doc_ids": doc_id_arr
        }

        if return_embeddings:
            result["embeddings"] = emb_arr

        # Automatically generate all plots and tables if an output directory is provided
        if output_dir is not None:
            save_evaluation_bundle(
                bundle=bundle, output_dir=output_dir, split=split,
                y_true=y_true, y_pred=y_pred, y_prob=y_prob,
                embeddings=emb_arr, label_names=self.label_names,
                history_rows=history_rows)

        self.model.train()
        return result

    def finetune(
        self, labeled_loader: DataLoader, unlabeled_loader: DataLoader, optimizer: AdamW,
        epochs: int, val_loader: DataLoader | None = None, log_interval: int = 20) -> dict[str, object]:
        """Executes the semi-supervised fine-tuning phase."""
        # Consistency Regularization & Pseudo-Labeling execute a dual-stream loop.
        # Labeled data drives standard cross-entropy. Unlabeled data is forwarded simultaneously.
        # Its weak predictions are aggressively filtered (sharpened, aligned, and dynamic-thresholded).
        # The surviving highly confident predictions act as "pseudo-labels" for a secondary
        # consistency loss, vastly improving generalization when labels are scarce.
        history: dict[str, list[float]] = {"epoch": [], "train_loss": [], "sup_loss": [], "ssl_loss": [], "pseudo_label_ratio": []}
        unlabeled_stream = cycle(unlabeled_loader)
        show_bar = self.accelerator.is_local_main_process
        epoch_bar = tqdm(range(1, epochs + 1), desc="Finetune", unit="ep", disable=not show_bar)

        for epoch in epoch_bar:
            self.model.train()
            total_loss, total_sup, total_ssl = 0.0, 0.0, 0.0
            total_selected, total_unlabeled = 0, 0
            threshold_track: list[float] = []
            step_bar = tqdm(
                labeled_loader, total=len(labeled_loader), desc=f"Finetune ep {epoch}/{epochs}",
                unit="batch", leave=False, dynamic_ncols=True, disable=not show_bar)

            for step, labeled_batch in enumerate(step_bar, start=1):
                unlabeled_batch = next(unlabeled_stream)
                with self.accelerator.accumulate(self.model):
                    optimizer.zero_grad()

                    # Supervised branch
                    _, labeled_logits, _ = self.forward(labeled_batch)
                    sup_loss = self.supervised_loss(labeled_logits, labeled_batch["labels"])

                    # Unlabeled branch: forward pass, selection, and consistency loss
                    _, unlabeled_logits, unlabeled_probs = self.forward(unlabeled_batch)
                    selected, pseudo_labels, thresholds, _ = self.pseudo_labeler.select(unlabeled_probs, epoch=epoch)
                    if selected.any():
                        ssl_loss = self.supervised_loss(unlabeled_logits[selected], pseudo_labels[selected])
                    else:
                        ssl_loss = unlabeled_logits.new_zeros(())

                    # Combine objectives
                    loss = sup_loss + float(self.cfg["lambda_ssl"]) * ssl_loss
                    self.accelerator.backward(loss)
                    self.clip()
                    optimizer.step()

                # Accumulate batch statistics
                loss_value = float(loss.detach().item())
                sup_value = float(sup_loss.detach().item())
                ssl_value = float(ssl_loss.detach().item())
                total_loss += loss_value
                total_sup += sup_value
                total_ssl += ssl_value
                total_selected += int(selected.sum().item())
                total_unlabeled += int(unlabeled_probs.size(0))
                threshold_track.append(float(thresholds.mean().item()))
                step_bar.set_postfix(
                    total=f"{loss_value:.4f}", sup=f"{sup_value:.4f}", ssl=f"{ssl_value:.4f}")

                if step % log_interval == 0 and show_bar:
                    tqdm.write(
                        f"[Finetune] epoch={epoch} step={step}/{len(labeled_loader)} "
                        f"total={loss_value:.4f} sup={sup_value:.4f} ssl={ssl_value:.4f}")

            step_bar.close()

            row: dict[str, float | int | str] = {
                "stage": "finetune", "epoch": epoch,
                "train_loss": total_loss / max(len(labeled_loader), 1),
                "sup_loss": total_sup / max(len(labeled_loader), 1),
                "ssl_loss": total_ssl / max(len(labeled_loader), 1),
                "pseudo_label_ratio": total_selected / max(total_unlabeled, 1),
                "pseudo_threshold_mean": float(np.mean(threshold_track)) if threshold_track else float("nan")
            }

            # Optional validation checkpointing mechanism
            if val_loader is not None:
                val_result = self.evaluate(
                    val_loader, split="val",
                    output_dir=self.output_dir / "artifacts" / "val",
                    history_rows=self.history + [row])
                row.update({f"val_{k}": v for k, v in val_result["metrics"].items()})

                score = float(val_result["metrics"].get(str(self.cfg["selection_metric"]), val_result["metrics"].get("macro_f1", 0.0)))
                if score > self.best_score:
                    self.best_score = score
                    self.save_checkpoint(epoch=epoch, best_score=score)

            history["epoch"].append(float(epoch))
            for k in ["train_loss", "sup_loss", "ssl_loss", "pseudo_label_ratio"]:
                history[k].append(float(row[k]))

            self.history.append(row)
            self.log_metrics({k: v for k, v in row.items() if isinstance(v, (float, int))}, epoch, "finetune")

            if show_bar:
                val_score = row.get(f"val_{self.cfg['selection_metric']}")
                summary = (
                    f"[Finetune] epoch={epoch}/{epochs} train_loss={row['train_loss']:.4f} "
                    f"sup={row['sup_loss']:.4f} ssl={row['ssl_loss']:.4f} "
                    f"pseudo_ratio={row['pseudo_label_ratio']:.3f}"
                )
                if isinstance(val_score, (int, float)):
                    summary += f" val_{self.cfg['selection_metric']}={float(val_score):.4f}"
                logger.info(summary)
            postfix = {"loss": f"{row['train_loss']:.4f}"}
            val_score = row.get(f"val_{self.cfg['selection_metric']}")
            if isinstance(val_score, (int, float)):
                postfix[f"val_{self.cfg['selection_metric']}"] = f"{float(val_score):.4f}"
            epoch_bar.set_postfix(**postfix)

        epoch_bar.close()
        # Export visualizations and history trails to filesystem
        save_frame(to_frame(self.history), self.output_dir / "artifacts" / "history.csv")
        plot_training_history(self.history, self.output_dir / "artifacts" / "training_curves.png")
        return {
            "history": history, "history_rows": self.history,
            "best_score": self.best_score, "best_checkpoint": str(self.best_checkpoint)
        }

    def train_full_pipeline(
        self, pretrain_loader: DataLoader, labeled_loader: DataLoader, unlabeled_loader: DataLoader,
        val_loader: DataLoader | None = None, test_loader: DataLoader | None = None,
        pretrain_epochs: int = 20, finetune_epochs: int = 10) -> dict[str, object]:
        """Orchestrates the entire end-to-end training lifecycle."""
        pretrain_optimizer = self.optimizer("pretrain")
        finetune_optimizer = self.optimizer("finetune")

        items: list[Any] = [
            self.model, pretrain_optimizer, finetune_optimizer,
            pretrain_loader, labeled_loader, unlabeled_loader]

        if val_loader is not None:
            items.append(val_loader)
        if test_loader is not None:
            items.append(test_loader)

        # Apply multi-device and mixed-precision wrapping
        prepared = self.accelerator.prepare(*items)
        self.model = prepared[0]
        pretrain_optimizer = prepared[1]
        finetune_optimizer = prepared[2]
        pretrain_loader = prepared[3]
        labeled_loader = prepared[4]
        unlabeled_loader = prepared[5]

        cursor = 6
        if val_loader is not None:
            val_loader = prepared[cursor]
            cursor += 1

        if test_loader is not None:
            test_loader = prepared[cursor]

        self.start_run()
        try:
            # Sequential execution of training stages
            pretrain_result = self.pretrain(pretrain_loader, pretrain_optimizer, epochs=pretrain_epochs)
            finetune_result = self.finetune(
                labeled_loader, unlabeled_loader, finetune_optimizer,
                epochs=finetune_epochs, val_loader=val_loader)

            # Post-training evaluation on best weights
            test_result = None
            if test_loader is not None:
                self.load_checkpoint()
                artifact_dir = self.output_dir / "artifacts" / "test"

                test_result = self.evaluate(
                    test_loader, split="test", return_embeddings=True,
                    output_dir=artifact_dir, history_rows=self.history)
                (artifact_dir / "test_metrics.json").write_text(json.dumps(test_result["metrics"], indent=2))

            # Sync all generated charts/csvs to MLflow
            if bool(self.cfg.get("use_mlflow")):
                mlflow.log_artifacts(str(self.output_dir / "artifacts"))
            # Sync all generated charts/csvs to W&B
            if bool(self.cfg.get("use_wandb")):
                wandb.save(str(self.output_dir / "artifacts" / "*"), base_path=str(self.output_dir))
            return {
                "pretrain": pretrain_result, "finetune": finetune_result,
                "test": test_result, "history_rows": self.history
            }
        finally:
            if bool(self.cfg.get("use_mlflow")):
                mlflow.end_run()
            if bool(self.cfg.get("use_wandb")):
                wandb.finish()
