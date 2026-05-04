from typing import Sequence
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, f1_score, log_loss, matthews_corrcoef,
    precision_recall_fscore_support, precision_score, recall_score, roc_auc_score)


def multiclass_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    y_prob: np.ndarray | None = None
) -> dict[str, float]:
    """
    Collects the main summary indicators used in multiclass evaluation.

    This centralizes the full evaluation summary so all stages use the same
    statistical definitions and remain directly comparable across experiments.
    """
    # Keep the overall summary in a single place so training, validation,
    # and testing always report the same quantities in the same format.
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred))
    }

    # Some indicators require confidence distributions rather than final labels.
    # These are optional because not every baseline exposes calibrated scores.
    if y_prob is not None:
        y_prob_safe = np.where(np.isfinite(y_prob), y_prob, 1.0 / y_prob.shape[1])
        y_prob_safe = y_prob_safe / np.maximum(y_prob_safe.sum(axis=1, keepdims=True), 1e-12)
        metrics["log_loss"] = float(log_loss(y_true, y_prob_safe, labels=np.arange(y_prob.shape[1])))
        try:
            metrics["macro_ovr_auroc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            metrics["macro_ovr_auroc"] = float("nan")

    return metrics


def per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    label_names: Sequence[str] | None = None
) -> pl.DataFrame:
    """
    Builds a per-category report with precision, recall, F1, and support.

    The report covers every category that appears either in the reference labels
    or in the predictions, so rare or degenerate cases are still visible.
    """
    # Use the union of observed references and predictions so the report still
    # exposes missing categories, over-predicted categories, or collapsed outputs.
    classes = np.unique(np.concatenate([y_true, y_pred]))

    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    # Convert explicitly to arrays so static type checkers understand that
    # the returned values are indexable sequences rather than scalar floats.
    precision_arr = np.asarray(precision_arr, dtype=np.float64)
    recall_arr = np.asarray(recall_arr, dtype=np.float64)
    f1_arr = np.asarray(f1_arr, dtype=np.float64)
    support_arr = np.asarray(support_arr, dtype=np.int64)

    rows: list[dict[str, int | float | str]] = []

    for idx, cls in enumerate(classes):
        class_id = int(cls)
        class_name = (
            label_names[class_id]
            if label_names is not None and class_id < len(label_names)
            else str(class_id)
        )

        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "precision": float(precision_arr[idx]),
            "recall": float(recall_arr[idx]),
            "f1": float(f1_arr[idx]),
            "support": int(support_arr[idx])
        })

    return pl.DataFrame(rows)


def prediction_table(
    doc_ids: Sequence[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None
) -> pl.DataFrame:
    """
    Builds a per-sample prediction table with confidence and correctness.

    This makes downstream error analysis easier by preserving one row per sample,
    including confidence and whether the final decision was correct.
    """
    # Keep a sample-level audit table so failure cases, calibration behavior,
    # and difficult examples can be inspected without recomputing predictions.
    confidence = (
        y_prob.max(axis=1)
        if y_prob is not None
        else np.full(len(y_true), np.nan, dtype=np.float32)
    )

    rows: list[dict[str, int | float]] = [{
        "doc_id": int(doc_id),
        "label": int(true_label),
        "prediction": int(pred_label),
        "confidence": float(conf_score),
        "correct": int(true_label == pred_label),
    } for doc_id, true_label, pred_label, conf_score in zip(doc_ids, y_true, y_pred, confidence)]

    return pl.DataFrame(rows)
