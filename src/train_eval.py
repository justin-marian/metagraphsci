import json
import math
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
from torch.optim.lr_scheduler import LambdaLR
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

Stage 1 — self-supervised contrastive pretraining.
Stage 2 — supervised + pseudo-labeled fine-tuning.
"""


class MetaGraphSciTrainerEval:
    """Two-stage trainer for MetaGraphSci with clean separation of concerns."""

    DEFAULTS = {
        "output_dir": "../out/",
        "mixed_precision": "no",
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,

        # HYPERPARAMETER IMPROVEMENT: lower LRs are safer for SciBERT.
        # 1e-4 / 5e-5 original values caused representation corruption in first
        # steps without warmup.  New values: 3e-5 pretrain, 2e-5 finetune.
        "pretrain_lr": 3e-5,
        "finetune_lr": 2e-5,
        "weight_decay": 0.01,

        # HYPERPARAMETER IMPROVEMENT: 6% warmup + cosine decay is the standard
        # BERT fine-tuning recipe.
        "lr_warmup_fraction": 0.06,

        "selection_metric": "macro_f1",
        "run_name": "baseline",

        "use_mlflow": False, "mlflow_experiment": "MetaGraphSci",
        "use_wandb":  False, "wandb_project":    "MetaGraphSci",

        "contrastive_temperature": 0.10,   # 0.07→0.10: slightly warmer for academic docs
        "metadata_negative_weight": 0.25,
        "ssl_text_dropout": 0.15,

        # HYPERPARAMETER IMPROVEMENT: reduce pseudo-label weight to 0.5.
        # Starting at 1.0 gave pseudo-labels equal weight to supervised signal
        # from the very first finetune step, before the model was reliable enough
        # to generate trustworthy pseudo-targets.
        "lambda_ssl": 0.5,
        "lambda_ssl_final": 0.5,
        "supervised_warmup_epochs": 5,
        "pseudo_ramp_epochs": 8,
        "min_pseudo_confidence": 0.0,

        "ablation_mode": "full",

        # HYPERPARAMETER IMPROVEMENT: see PseudoLabeler for individual rationales.
        "pseudo_label": {
            "beta": 0.80,            # 0.95→0.80 — less restrictive, more pseudo-labels
            "warmup_epochs": 3,      # 1→3  — wait for a more stable model
            "min_per_class": 0,
            "temperature": 0.5,      # 1.0→0.5 — sharper = more reliable targets
            "ema_momentum": 0.95,    # 0.90→0.95 — smoother curriculum
            "distributionalignment": True,
        },

        # HYPERPARAMETER IMPROVEMENT: 0.1 label smoothing works well with the
        # NormalizedCosineClassifier because the cosine logits are inherently
        # bounded [-scale, +scale] and can cause overconfident CE gradients.
        "label_smoothing": 0.1,
    }

    def __init__(
        self, model: nn.Module, citation_graph: Any,
        neighbor_cache: Mapping[int, set[int]] | None = None,
        config: Mapping[str, object] | None = None,
        label_names: Sequence[str] | None = None,
        labeled_class_prior: Sequence[float] | None = None,
    ) -> None:
        self.cfg = {**self.DEFAULTS, **dict(config or {})}
        pseudo_cfg = {**self.DEFAULTS["pseudo_label"],
                      **dict(self.cfg.get("pseudo_label", {}))}
        self.cfg["pseudo_label"] = pseudo_cfg

        self.model         = model
        self.graph         = citation_graph
        self.neighbor_cache = {int(k): set(map(int, v))
            for k, v in (neighbor_cache or {}).items()}
        self.label_names   = list(label_names) if label_names is not None else None
        self.supported_labels = [i for i, p in enumerate(labeled_class_prior or []) if p > 0]

        self.output_dir      = Path(str(self.cfg["output_dir"]))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint = self.output_dir / "checkpoints" / "best_model.pt"
        self.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)

        setup_global_logger(self.output_dir)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=int(self.cfg["gradient_accumulation_steps"]),
            mixed_precision=str(self.cfg["mixed_precision"]))
        self.device = self.accelerator.device

        pseudo_cfg_for_init = {k: v for k, v in pseudo_cfg.items() if k != "target_prior"}
        self.contrastive_loss = NeighborhoodAwareContrastiveLoss(
            float(self.cfg["contrastive_temperature"]),
            float(self.cfg["metadata_negative_weight"]))

        # HYPERPARAMETER IMPROVEMENT: label_smoothing=0.1 reduces overconfidence
        # that is otherwise amplified by the cosine classifier's scale factor.
        self.supervised_loss = nn.CrossEntropyLoss(
            label_smoothing=float(self.cfg.get("label_smoothing", 0.1)))

        self.pseudo_labeler = PseudoLabeler(
            target_prior=labeled_class_prior, **pseudo_cfg_for_init)

        self.history: list[dict[str, float | int | str]] = []
        self.best_score = float("-inf")
        self.ablation_mode = str(self.cfg["ablation_mode"])

    def pseudo_weight(self, epoch: int) -> float:
        """Return the pseudo-label loss weight for the current fine-tuning epoch."""
        warmup = int(self.cfg.get("supervised_warmup_epochs", 0))
        if epoch <= warmup:
            return 0.0

        ramp_epochs = max(1, int(self.cfg.get("pseudo_ramp_epochs", 1)))
        final = float(self.cfg.get("lambda_ssl_final", self.cfg.get("lambda_ssl", 0.0)))
        progress = min(1.0, float(epoch - warmup) / float(ramp_epochs))
        return final * progress

    def _make_optimizer(self, stage: str) -> AdamW:
        lr = float(self.cfg["pretrain_lr"] if stage == "pretrain" else self.cfg["finetune_lr"])
        fused = torch.cuda.is_available()
        return AdamW(self.model.parameters(), lr=lr, weight_decay=float(self.cfg["weight_decay"]), fused=fused)

    def _make_scheduler(self, optimizer: AdamW, total_steps: int) -> LambdaLR:
        """Linear warmup + cosine decay — the standard BERT fine-tuning recipe.
        Starting SciBERT
        at lr=1e-4 from step 0 corrupts pre-trained representations in the first
        few gradient steps, making the model unable to learn regardless of any
        other fix applied.
        """
        warmup_steps = max(1, int(total_steps * float(self.cfg["lr_warmup_fraction"])))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def extract_context_tensors(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {k: v for k, v in batch.items()
                if k.startswith("context_") and isinstance(v, Tensor)}

    def forward(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            venue_ids=batch["venue_ids"],
            publisher_ids=batch["publisher_ids"],
            author_ids=batch["author_ids"],
            years=batch["years"],
            ablation_mode=self.ablation_mode,
            **self.extract_context_tensors(batch))

    def embeddings(self, batch: Mapping[str, Tensor]) -> Tensor:
        return self.model.get_embeddings(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            venue_ids=batch["venue_ids"],
            publisher_ids=batch["publisher_ids"],
            author_ids=batch["author_ids"],
            years=batch["years"],
            ablation_mode=self.ablation_mode,
            **self.extract_context_tensors(batch))

    def augment_batch_for_ssl(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Masks tokens in the anchor view to create a corrupted positive pair."""
        augmented = {k: v.clone() if isinstance(v, Tensor) else v for k, v in batch.items()}
        input_ids      = augmented["input_ids"]
        attention_mask = augmented["attention_mask"].bool()
        probability    = float(self.cfg.get("ssl_text_dropout", 0.15))

        if probability <= 0.0:
            return augmented

        dropout_mask = (torch.rand_like(input_ids.float()) < probability) & attention_mask
        special      = (input_ids == 0) | (input_ids == 101) | (input_ids == 102)
        dropout_mask = dropout_mask & ~special
        augmented["input_ids"] = input_ids.masked_fill(dropout_mask, 103)
        return augmented

    def metadata_affinity(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Boolean mask: True when two docs share venue, publisher, or ±2 year window."""
        venue     = batch["venue_ids"]
        publisher = batch["publisher_ids"]
        years     = batch["years"].view(-1)

        same_venue     = venue.unsqueeze(1)  == venue.unsqueeze(0)
        same_publisher = publisher.unsqueeze(1) == publisher.unsqueeze(0)
        # Raw ±2-year window instead of 2.0/26.0.
        # The old divisor assumed years are normalised over a 26-year range,
        # but MetadataEncoder receives raw year values (e.g. 2018, 2021).
        # The 2.0/26.0 threshold (~0.077) would only match papers published
        # within ~2 months of each other, making close_year almost always False.
        close_year     = (years.unsqueeze(1) - years.unsqueeze(0)).abs() <= 2.0

        affinity = same_venue | same_publisher | close_year
        affinity.fill_diagonal_(False)
        return affinity

    def build_positive_mask(
        self, batch: Mapping[str, Tensor], metadata_affinity: Tensor
    ) -> Tensor:
        """Graph-adjacent AND metadata-compatible pairs only.

        BUG FIXED: the original fallback accepted metadata-only pairs (same
        venue / publisher / year) when no graph-adjacent document appeared in
        the batch.  Publication-venue overlap is not semantic similarity — a
        NeurIPS batch spans NLP, CV, RL, and theory.  Pulling those pairs
        together during contrastive pretraining trained the encoder to cluster
        by venue rather than topic, actively harming classification accuracy.

        Fix: when no graph-neighbour pair is present, leave the row empty.
        The contrastive loss already handles this by falling back to the
        SimCLR diagonal (anchor vs its own masked-token augmentation), which
        is always a safe positive.
        """
        doc_ids    = batch["doc_id"].detach().cpu().tolist()
        batch_size = len(doc_ids)
        mask       = torch.zeros((batch_size, batch_size),
            dtype=torch.bool, device=metadata_affinity.device)

        for i, doc_id in enumerate(doc_ids):
            neighbors  = self.neighbor_cache.get(int(doc_id), set())
            # Graph-adjacency alone as the positive criterion.
            # The old code required BOTH adjacency AND metadata compatibility,
            # whose intersection is almost always empty in a mini-batch, causing
            # the contrastive loss to degenerate to the SimCLR diagonal only and
            # the model to never learn from graph topology.
            # Metadata is still used for negative *softening* inside the loss.
            candidates = [
                j for j, other_id in enumerate(doc_ids)
                if i != j and int(other_id) in neighbors]
            for j in candidates:
                mask[i, j] = True

        return mask

    def clip(self) -> None:
        self.accelerator.clip_grad_norm_(
            self.model.parameters(), float(self.cfg["max_grad_norm"]))

    def gather(self, tensor: Tensor) -> Tensor:
        return self.accelerator.gather_for_metrics(tensor).detach().cpu()

    def neighborhoods(self, batch_doc_ids: Tensor) -> list[set[int]]:
        return [self.neighbor_cache.get(int(d), set())
                for d in batch_doc_ids.detach().cpu().tolist()]

    def save_checkpoint(self, **extra: Any) -> None:
        """Saves model weights, pseudo-labeler EMA, and scheduler state."""
        payload = {
            "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
            "pseudo_labeler":   self.pseudo_labeler.labeler_state_dict(),
            **extra,
        }
        torch.save(payload, self.best_checkpoint)

    def load_checkpoint(self) -> None:
        """Rehydrates model weights and pseudo-labeler EMA from disk."""
        if self.best_checkpoint.exists():
            state = torch.load(self.best_checkpoint, map_location="cpu")
            self.accelerator.unwrap_model(self.model).load_state_dict(
                state["model_state_dict"])
            if "pseudo_labeler" in state:
                self.pseudo_labeler.load_labeler_state_dict(state["pseudo_labeler"])

    def log_metrics(self, metrics: Mapping[str, float], step: int, prefix: str) -> None:
        clean = {f"{prefix}/{k}": float(v)
                for k, v in metrics.items()
                if v is not None and np.isfinite(v)}
        if not clean:
            return
        if bool(self.cfg.get("use_mlflow")):
            mlflow.log_metrics(clean, step=step)
        if bool(self.cfg.get("use_wandb")):
            wandb.log(clean, step=step)

    def start_run(self) -> None:
        flat = {k: v for k, v in self.cfg.items()
                if isinstance(v, (int, float, str, bool, dict))}
        if bool(self.cfg.get("use_mlflow")):
            mlflow.set_experiment(str(self.cfg["mlflow_experiment"]))
            mlflow.start_run(run_name=str(self.cfg["run_name"]))
            mlflow.log_params({k: v for k, v in self.cfg.items() if isinstance(v, (int, float, str, bool))})
        if bool(self.cfg.get("use_wandb")):
            wandb.init(project=str(self.cfg["wandb_project"]),
                name=str(self.cfg["run_name"]),
                config=flat, reinit=True)

    def pretrain(
        self, loader: DataLoader, optimizer: AdamW,
        scheduler: LambdaLR, epochs: int, log_interval: int = 20,
    ) -> dict[str, list[float]]:
        """Self-supervised contrastive pretraining."""
        history  = {"epoch": [], "train_loss": []}
        show_bar = self.accelerator.is_local_main_process
        epoch_bar = tqdm(range(1, epochs + 1), desc="Pretrain", unit="ep", disable=not show_bar)

        for epoch in epoch_bar:
            self.model.train()
            total_loss, total_steps = 0.0, 0.0
            step_bar = tqdm(
                loader, total=len(loader),
                desc=f"Pretrain ep {epoch}/{epochs}",
                unit="batch", leave=False,
                dynamic_ncols=True, disable=not show_bar)

            for step, batch in enumerate(step_bar, start=1):
                with self.accelerator.accumulate(self.model):
                    optimizer.zero_grad()

                    anchor         = self.embeddings(batch)
                    positive_batch = self.augment_batch_for_ssl(batch)
                    positive       = self.embeddings(positive_batch)

                    metadata_affinity = self.metadata_affinity(batch)
                    positive_mask     = self.build_positive_mask(batch, metadata_affinity)

                    loss = self.contrastive_loss(
                        anchor, positive, batch["doc_id"],
                        self.neighborhoods(batch["doc_id"]),
                        metadata_affinity=metadata_affinity,
                        positive_mask=positive_mask)

                    self.accelerator.backward(loss)
                    self.clip()
                    optimizer.step()
                    scheduler.step()

                loss_value  = float(loss.detach().item())
                total_loss += loss_value
                total_steps += 1
                step_bar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if step % log_interval == 0 and show_bar:
                    tqdm.write(
                        f"[Pretrain] ep={epoch} step={step}/{len(loader)} "
                        f"loss={loss_value:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}")

            step_bar.close()
            avg = total_loss / max(total_steps, 1)
            row = {"stage": "pretrain", "epoch": epoch, "train_loss": avg}

            if show_bar:
                logger.info(f"[Pretrain] epoch={epoch}/{epochs} avg_loss={avg:.4f}")
            epoch_bar.set_postfix(avg_loss=f"{avg:.4f}")

            history["epoch"].append(epoch)
            history["train_loss"].append(avg)
            self.history.append(row)
            self.log_metrics({"train_loss": avg}, epoch, "pretrain")

        epoch_bar.close()
        return history

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader, split: str = "val",
        return_embeddings: bool = False,
        output_dir: Path | None = None,
        history_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Full-pass evaluation with optional artifact export."""
        self.model.eval()

        doc_ids, labels, preds, probs, embeddings = [], [], [], [], []
        total_loss, total_steps = 0.0, 0.0
        show_bar = self.accelerator.is_local_main_process
        eval_bar = tqdm(loader, total=len(loader),
                        desc=f"Eval[{split}]", unit="batch",
                        leave=False, dynamic_ncols=True, disable=not show_bar)

        for batch in eval_bar:
            batch = {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            z, logits, batch_probs = self.forward(batch)
            loss = self.supervised_loss(logits, batch["labels"])
            total_loss  += float(loss.detach().item())
            total_steps += 1

            doc_ids.append(self.gather(batch["doc_id"]).detach().cpu())
            labels.append(self.gather(batch["labels"]).detach().cpu())
            preds.append(self.gather(logits.argmax(dim=1)).detach().cpu())
            probs.append(self.gather(batch_probs).detach().cpu())

            if return_embeddings or output_dir is not None:
                embeddings.append(self.gather(z).detach().cpu())

            eval_bar.set_postfix(loss=f"{total_loss / max(total_steps, 1):.4f}")

        eval_bar.close()

        y_true     = torch.cat(labels).numpy()
        y_pred     = torch.cat(preds).numpy()
        y_prob     = torch.cat(probs).numpy()
        doc_id_arr = torch.cat(doc_ids).numpy()
        emb_arr    = torch.cat(embeddings).numpy() if embeddings else None

        bundle = evaluate_predictions(
            y_true=y_true, y_pred=y_pred, y_prob=y_prob,
            doc_ids=doc_id_arr, label_names=self.label_names,
            supported_labels=self.supported_labels or None)

        result: dict[str, Any] = {
            "split": split,
            "metrics": {"loss": total_loss / max(total_steps, 1), **bundle["metrics"]},
            "bundle": bundle,
            "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
            "doc_ids": doc_id_arr,
        }

        if return_embeddings:
            result["embeddings"] = emb_arr

        if output_dir is not None:
            save_evaluation_bundle(
                bundle=bundle, output_dir=output_dir, split=split,
                y_true=y_true, y_pred=y_pred, y_prob=y_prob,
                embeddings=emb_arr, label_names=self.label_names,
                history_rows=history_rows)

        self.model.train()
        return result

    def finetune(
        self, labeled_loader: DataLoader, unlabeled_loader: DataLoader,
        optimizer: AdamW, scheduler: LambdaLR, epochs: int,
        val_loader: DataLoader | None = None,
        log_interval: int = 20,
    ) -> dict[str, object]:
        """Semi-supervised fine-tuning with pseudo-labeling."""
        history: dict[str, list[float]] = {
            "epoch": [], "train_loss": [], "sup_loss": [],
            "ssl_loss": [], "pseudo_label_ratio": [], "pseudo_loss_weight": []}

        unlabeled_stream = cycle(unlabeled_loader)
        show_bar  = self.accelerator.is_local_main_process
        epoch_bar = tqdm(range(1, epochs + 1), desc="Finetune", unit="ep", disable=not show_bar)

        for epoch in epoch_bar:
            self.model.train()
            total_loss, total_sup, total_ssl = 0.0, 0.0, 0.0
            total_selected, total_unlabeled  = 0, 0
            threshold_track: list[float]     = []

            step_bar = tqdm(
                labeled_loader, total=len(labeled_loader),
                desc=f"Finetune ep {epoch}/{epochs}",
                unit="batch", leave=False,
                dynamic_ncols=True, disable=not show_bar)

            for step, labeled_batch in enumerate(step_bar, start=1):
                unlabeled_batch = next(unlabeled_stream)

                with self.accelerator.accumulate(self.model):
                    optimizer.zero_grad()

                    _, labeled_logits, _ = self.forward(labeled_batch)
                    sup_loss = self.supervised_loss(
                        labeled_logits, labeled_batch["labels"])

                    _, unlabeled_logits, unlabeled_probs = self.forward(unlabeled_batch)
                    with torch.no_grad():
                        selected, pseudo_labels, thresholds, adjusted_probs = (
                            self.pseudo_labeler.select(unlabeled_probs.detach(), epoch=epoch))
                        min_conf = float(self.cfg.get("min_pseudo_confidence", 0.0))
                        if min_conf > 0.0:
                            confidence = adjusted_probs.max(dim=1).values
                            selected = selected & (confidence >= min_conf)

                    if selected.any():
                        ssl_loss = self.supervised_loss(
                            unlabeled_logits[selected], pseudo_labels[selected])
                    else:
                        ssl_loss = unlabeled_logits.new_zeros(())

                    ssl_weight = self.pseudo_weight(epoch)
                    loss = sup_loss + ssl_weight * ssl_loss
                    self.accelerator.backward(loss)
                    self.clip()
                    optimizer.step()
                    scheduler.step()

                loss_value = float(loss.detach().item())
                sup_value  = float(sup_loss.detach().item())
                ssl_value  = float(ssl_loss.detach().item())
                total_loss += loss_value
                total_sup  += sup_value
                total_ssl  += ssl_value
                total_selected  += int(selected.sum().item())
                total_unlabeled += int(unlabeled_probs.size(0))
                threshold_track.append(float(thresholds.mean().item()))

                step_bar.set_postfix(
                    total=f"{loss_value:.4f}", sup=f"{sup_value:.4f}",
                    ssl=f"{ssl_value:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if step % log_interval == 0 and show_bar:
                    tqdm.write(
                        f"[Finetune] epoch={epoch} step={step}/{len(labeled_loader)} "
                        f"total={loss_value:.4f} sup={sup_value:.4f} "
                        f"ssl={ssl_value:.4f}")

            step_bar.close()

            n_steps = max(len(labeled_loader), 1)
            row: dict[str, float | int | str] = {
                "stage": "finetune", "epoch": epoch,
                "train_loss":          total_loss / n_steps,
                "sup_loss":            total_sup  / n_steps,
                "ssl_loss":            total_ssl  / n_steps,
                "pseudo_label_ratio":  total_selected / max(total_unlabeled, 1),
                "pseudo_loss_weight": self.pseudo_weight(epoch),
                "pseudo_threshold_mean": (float(np.mean(threshold_track))
                if threshold_track else float("nan")),
            }

            if val_loader is not None:
                val_result = self.evaluate(
                    val_loader, split="val",
                    output_dir=self.output_dir / "artifacts" / "val",
                    history_rows=self.history + [row])
                row.update({f"val_{k}": v
                            for k, v in val_result["metrics"].items()})
                score = float(val_result["metrics"].get(
                    str(self.cfg["selection_metric"]),
                    val_result["metrics"].get("macro_f1", 0.0)))
                if score > self.best_score:
                    self.best_score = score
                    self.save_checkpoint(epoch=epoch, best_score=score)

            history["epoch"].append(float(epoch))
            for k in ["train_loss", "sup_loss", "ssl_loss", "pseudo_label_ratio", "pseudo_loss_weight"]:
                history[k].append(float(row[k]))

            self.history.append(row)
            self.log_metrics(
                {k: v for k, v in row.items() if isinstance(v, (float, int))},
                epoch, "finetune")

            if show_bar:
                val_score = row.get(f"val_{self.cfg['selection_metric']}")
                summary   = (
                    f"[Finetune] epoch={epoch}/{epochs} "
                    f"train_loss={row['train_loss']:.4f} "
                    f"sup={row['sup_loss']:.4f} ssl={row['ssl_loss']:.4f} "
                    f"pseudo_ratio={row['pseudo_label_ratio']:.3f} "
                    f"pseudo_w={row['pseudo_loss_weight']:.3f}")
                if isinstance(val_score, (int, float)):
                    summary += f" val_{self.cfg['selection_metric']}={float(val_score):.4f}"
                logger.info(summary)

            postfix   = {"loss": f"{row['train_loss']:.4f}"}
            val_score = row.get(f"val_{self.cfg['selection_metric']}")
            if isinstance(val_score, (int, float)):
                postfix[f"val_{self.cfg['selection_metric']}"] = f"{float(val_score):.4f}"
            epoch_bar.set_postfix(**postfix)

        epoch_bar.close()
        save_frame(to_frame(self.history), self.output_dir / "artifacts" / "history.csv")
        plot_training_history(self.history,
                self.output_dir / "artifacts" / "training_curves.png")
        return {
            "history": history, "history_rows": self.history,
            "best_score": self.best_score,
            "best_checkpoint": str(self.best_checkpoint),
        }

    def train_full_pipeline(
        self,
        pretrain_loader: DataLoader,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        pretrain_epochs: int = 20,
        finetune_epochs: int = 10
    ) -> dict[str, object]:
        """Orchestrates pretraining → fine-tuning → test evaluation."""

        # FIX #3: prepare model and loaders first — no optimizers yet.
        # Creating both AdamW optimizers simultaneously doubled peak optimizer
        # memory (each holds 2x |params| moment tensors) and caused fused=True
        # AdamW to receive CPU parameters before prepare moved the model to GPU.
        # Solution: prepare loaders + model together, then create and prepare
        # each optimizer just before its own stage so only one set of moment
        # tensors is ever live at a time.
        items: list[Any] = [self.model, pretrain_loader, labeled_loader, unlabeled_loader]
        if val_loader  is not None: 
            items.append(val_loader)
        if test_loader is not None: 
            items.append(test_loader)

        prepared         = self.accelerator.prepare(*items)
        self.model       = prepared[0]
        pretrain_loader  = prepared[1]
        labeled_loader   = prepared[2]
        unlabeled_loader = prepared[3]

        cursor = 4
        if val_loader  is not None:
            val_loader  = prepared[cursor]; cursor += 1
        if test_loader is not None:
            test_loader = prepared[cursor]

        # Build pretrain optimizer + scheduler AFTER model is on the right device
        pretrain_optimizer = self._make_optimizer("pretrain")
        pretrain_optimizer = self.accelerator.prepare(pretrain_optimizer)
        pretrain_total     = max(1, len(pretrain_loader)) * pretrain_epochs
        pretrain_scheduler = self._make_scheduler(pretrain_optimizer, pretrain_total)

        # finetune_total is pre-computed here so the variable exists in scope;
        # the actual optimizer is created after pretraining completes (see below)
        # to avoid holding two optimizer states simultaneously.
        finetune_total = max(1, len(labeled_loader)) * finetune_epochs

        self.start_run()
        try:
            pretrain_result = self.pretrain(
                pretrain_loader, pretrain_optimizer, pretrain_scheduler,
                epochs=pretrain_epochs)

            # Release pretrain optimizer state before
            # allocating the finetune optimizer so only one AdamW moment buffer
            # (2x |params|) is live at a time.
            del pretrain_optimizer, pretrain_scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            finetune_optimizer = self._make_optimizer("finetune")
            finetune_optimizer = self.accelerator.prepare(finetune_optimizer)
            finetune_scheduler = self._make_scheduler(finetune_optimizer, finetune_total)

            finetune_result = self.finetune(
                labeled_loader, unlabeled_loader,
                finetune_optimizer, finetune_scheduler,
                epochs=finetune_epochs, val_loader=val_loader)

            test_result = None
            if test_loader is not None:
                self.load_checkpoint()
                artifact_dir = self.output_dir / "artifacts" / "test"
                test_result  = self.evaluate(
                    test_loader, split="test", return_embeddings=True,
                    output_dir=artifact_dir, history_rows=self.history)
                (artifact_dir / "test_metrics.json").write_text(
                    json.dumps(test_result["metrics"], indent=2))

            if bool(self.cfg.get("use_mlflow")):
                mlflow.log_artifacts(str(self.output_dir / "artifacts"))
            if bool(self.cfg.get("use_wandb")):
                wandb.save(str(self.output_dir / "artifacts" / "*"), base_path=str(self.output_dir))

            return {
                "pretrain": pretrain_result,
                "finetune": finetune_result,
                "test":     test_result,
                "history_rows": self.history,
            }
        finally:
            if bool(self.cfg.get("use_mlflow")): mlflow.end_run()
            if bool(self.cfg.get("use_wandb")):  wandb.finish()
