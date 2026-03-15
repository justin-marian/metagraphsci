import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

from metrics import multiclass_metrics, per_class_metrics, prediction_table
from plotting import (
    plot_calibration, plot_class_support,
    plot_confidence_histogram, plot_confusion,
    plot_embedding_projection, plot_per_class_f1,
    plot_pseudo_label_ratio, plot_training_history)
from utils import ensure_dir, save_frame


def evaluate_predictions(
    y_true: Sequence[int], y_pred: Sequence[int], y_prob: np.ndarray | None = None,
    doc_ids: Sequence[int] | None = None, label_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Bundle aggregate metrics, class reports, and prediction rows into one evaluation payload."""
    # Canonical Array Conversion
    # Convert all incoming sequences to stable NumPy arrays once at the evaluation
    # boundary so the downstream metric and reporting functions operate on a
    # consistent numerical format regardless of whether the caller passed Python
    # lists, tensors converted earlier, or NumPy arrays directly.
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = None if y_prob is None else np.asarray(y_prob)
    doc_id_arr = np.arange(len(y_true_arr)) if doc_ids is None else np.asarray(doc_ids)

    return {
        "metrics": multiclass_metrics(y_true_arr, y_pred_arr, y_prob_arr),
        "per_class": per_class_metrics(y_true_arr, y_pred_arr, label_names),
        "predictions": prediction_table(doc_id_arr, y_true_arr, y_pred_arr, y_prob_arr),
    }


def save_evaluation_bundle(
    bundle: Mapping[str, Any], output_dir: str | Path,
    split: str, y_true: Sequence[int], y_pred: Sequence[int], y_prob: np.ndarray | None = None,
    embeddings: np.ndarray | None = None, label_names: Sequence[str] | None = None, 
    history_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Path]:
    """Save all applicable evaluation tables and plots for one dataset split."""
    # Structured Evaluation Artifact Layout
    # Keep every output path deterministic and split-scoped so experiment folders
    # remain easy to inspect, compare, and version. Allow training code to call the 
    # same export function without duplicating filenames.
    out_dir = ensure_dir(output_dir)

    paths: dict[str, Path] = {
        "metrics_json": out_dir / f"{split}_metrics.json",
        "per_class_csv": out_dir / f"{split}_per_class.csv",
        "predictions_csv": out_dir / f"{split}_predictions.csv",
        "confusion_png": out_dir / f"{split}_confusion.png",
        "class_support_png": out_dir / f"{split}_class_support.png",
        "per_class_f1_png": out_dir / f"{split}_per_class_f1.png",
        "confidence_png": out_dir / f"{split}_confidence.png"
    }

    # Core Tabular Exports
    # Persist the scalar metrics, class breakdown, and row-wise predictions first.
    # These tables are the canonical machine-readable outputs used later for
    # reporting, debugging, and cross-run aggregation.
    paths["metrics_json"].write_text(json.dumps(bundle["metrics"], indent=2))
    save_frame(bundle["per_class"], paths["per_class_csv"])
    save_frame(bundle["predictions"], paths["predictions_csv"])

    # Core Diagnostic Plots
    # These plots summarize class balance, per-class effectiveness, prediction
    # confidence, and normalized confusion structure for the current split.
    plot_confusion(
        y_true, y_pred, label_names,
        paths["confusion_png"], normalize=True)
    plot_class_support(bundle["per_class"], paths["class_support_png"])
    plot_per_class_f1(bundle["per_class"], paths["per_class_f1_png"])
    plot_confidence_histogram(bundle["predictions"], paths["confidence_png"])

    # Probability-aware Diagnostics
    # Calibration only makes sense when the model emits probability vectors.
    if y_prob is not None:
        paths["calibration_png"] = out_dir / f"{split}_calibration.png"
        plot_calibration(y_true, y_prob, paths["calibration_png"])

    # Embedding Space Visualization
    # Projection plots are optional because not every evaluation stage stores
    # penultimate features, but when available they provide a useful qualitative
    # view of class separation and representation quality.
    if embeddings is not None:
        paths["embedding_png"] = out_dir / f"{split}_umap.png"
        plot_embedding_projection(
            embeddings, y_true, paths["embedding_png"],
            label_names=label_names, method="umap")

    # Training Dynamics Diagnostics
    # History-based plots are global run artifacts rather than split-specific model
    # outputs, but saving them here keeps the evaluation bundle self-contained.
    if history_rows:
        paths["history_png"] = out_dir / "training_curves.png"
        plot_training_history(history_rows, paths["history_png"])
        paths["pseudo_ratio_png"] = out_dir / "pseudo_label_ratio.png"
        plot_pseudo_label_ratio(history_rows, paths["pseudo_ratio_png"])

    return paths
