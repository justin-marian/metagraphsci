from typing import Sequence
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, 
    cohen_kappa_score, f1_score, log_loss, matthews_corrcoef,
    precision_recall_fscore_support, precision_score, recall_score, roc_auc_score)


def to_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    """Aggregate metrics used for multiclass classification."""
    # Aggregate Evaluation Summary
    # Keep all top-level classification metrics in one function so training,
    # validation, and testing all use the exact same statistical definitions.
    # This avoids metric drift across scripts and guarantees that exported
    # experiment tables remain directly comparable.
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)), "cohen_kappa": float(cohen_kappa_score(y_true, y_pred))
    }

    # Probability-aware Metrics
    # Metrics like log-loss and AUROC require calibrated class probabilities
    # rather than hard argmax predictions. These are optional because some
    # baselines may only emit labels.
    if y_prob is not None:
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=np.arange(y_prob.shape[1])))
        try:
            metrics["macro_ovr_auroc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            metrics["macro_ovr_auroc"] = float("nan")

    return metrics


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: Sequence[str] | None = None) -> pl.DataFrame:
    """Build a class-level report with precision, recall, F1, and support."""
    # Explicit Class Coverage
    # Use the union of true and predicted labels so the report also captures
    # degenerate cases where a class is predicted but absent in the ground truth,
    # or present in the truth but never predicted.
    classes = np.unique(np.concatenate([y_true, y_pred]))

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    rows: list[dict[str, int | float | str]] = []

    for idx, cls in enumerate(classes):
        class_id = int(cls)
        class_name = (
            label_names[class_id]
            if label_names is not None and class_id < len(label_names)
            else str(class_id))

        rows.append({
            "class_id": class_id, "class_name": class_name,
            "precision": float(precision[idx]), 
            "recall": float(recall[idx]), 
            "f1": float(f1[idx]),
            "support": int(support[idx])
        })
    return to_frame(rows)


def prediction_table(doc_ids: Sequence[int], y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> pl.DataFrame:
    """Build a per-document prediction table with confidence and correctness."""
    # Prediction Audit Table
    # Store one row per document so downstream analysis can inspect failure cases,
    # calibration issues, hard examples, and semi-supervised pseudo-label quality
    # without recomputing predictions from raw tensors.
    confidence = (
        y_prob.max(axis=1) if y_prob is not None
        else np.full(len(y_true), np.nan, dtype=np.float32))

    rows: list[dict[str, int | float]] = [{
        "doc_id": int(doc_id),
        "label": int(true_label),
        "prediction": int(pred_label),
        "confidence": float(conf_score), 
        "correct": int(true_label == pred_label)
    } for doc_id, true_label, pred_label, conf_score in zip(doc_ids, y_true, y_pred, confidence)]
    return to_frame(rows)
