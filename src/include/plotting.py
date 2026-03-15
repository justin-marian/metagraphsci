from pathlib import Path
from typing import Any, Mapping, Sequence
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid")


def plot_confusion(
    y_true: Sequence[int], y_pred: Sequence[int],
    label_names: Sequence[str] | None,
    output_path: str | Path, normalize: bool = True) -> Path:
    """Plot and save a formatted confusion matrix."""
    # Confusion Structure Visualization
    # A confusion matrix remains the most direct way to inspect which classes
    # the model systematically mixes together. Optional row-wise normalization
    # makes the plot comparable even when the class distribution is imbalanced.
    out_path = Path(output_path)
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(np.float64)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    ticks = list(label_names) if label_names is not None else [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True,
        fmt=".2f" if normalize else "d", cmap="Blues",
        xticklabels=ticks, yticklabels=ticks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Normalized confusion matrix" if normalize else "Confusion matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_training_history(
    history_rows: Sequence[Mapping[str, Any]],
    output_path: str | Path, metrics: Sequence[str] | None = None) -> Path:
    """Plot selected training and validation metrics across epochs."""
    out_path = Path(output_path)

    selected_metrics = list(metrics or [
        "train_loss", 
        "val_loss", "val_accuracy", "val_macro_f1",
        "val_balanced_accuracy",
    ])

    available_metrics = [
        metric for metric in selected_metrics
        if any(metric in row and row[metric] is not None for row in history_rows)
    ]

    if not available_metrics:
        raise ValueError("No requested metrics found in history_rows.")

    plt.figure(figsize=(10, 6))

    for metric in available_metrics:
        xs = [row["epoch"] for row in history_rows if row.get(metric) is not None]
        ys = [row[metric] for row in history_rows if row.get(metric) is not None]
        sns.lineplot(x=xs, y=ys, label=metric)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training and evaluation history")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_class_support(per_class: pl.DataFrame, output_path: str | Path) -> Path:
    """Plot the number of samples available for each class."""
    # Class Imbalance Overview
    # Sorting the bars by support makes long-tail datasets easier to inspect and
    # immediately reveals whether poor performance may be driven by underrepresented classes.
    out_path = Path(output_path)
    frame = per_class.to_pandas().sort_values("support", ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=frame, x="class_name", y="support")
    plt.xticks(rotation=45, ha="right")
    plt.title("Class support")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_per_class_f1(per_class: pl.DataFrame, output_path: str | Path) -> Path:
    """Plot the F1 score computed independently for each class."""
    out_path = Path(output_path)
    frame = per_class.to_pandas().sort_values("f1", ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=frame, x="class_name", y="f1")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Per-class F1")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_confidence_histogram(predictions: pl.DataFrame, output_path: str | Path) -> Path:
    """Plot confidence distributions for correct and incorrect predictions."""
    # Confidence Separation Analysis
    # Comparing confidence distributions between correct and incorrect predictions
    # helps reveal whether the classifier is overconfident, underconfident, or well separated.
    out_path = Path(output_path)
    frame = predictions.to_pandas()

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=frame, x="confidence", hue="correct", stat="density",
        common_norm=False, bins=20, element="step")
    plt.title("Confidence by correctness")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_calibration(
    y_true: Sequence[int], y_prob: np.ndarray,
    output_path: str | Path, n_bins: int = 10) -> Path:
    """Plot a reliability diagram for confidence calibration analysis."""
    # Confidence Calibration
    # Good classifiers should assign confidence values that match observed
    # correctness frequency. This plot compares predicted confidence with
    # empirical accuracy after binning.
    out_path = Path(output_path)
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    pred_labels = y_prob_arr.argmax(axis=1)
    confidence = y_prob_arr.max(axis=1)
    correctness = (pred_labels == y_true_arr).astype(int)

    frac_pos, mean_pred = calibration_curve(correctness, confidence, n_bins=n_bins, strategy="uniform")

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title("Reliability diagram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_pseudo_label_ratio(history_rows: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Plot the accepted pseudo-label ratio across fine-tuning epochs."""
    # Semi-Supervised Uptake Monitoring
    # Pseudo-label acceptance is a critical diagnostic in low-label training.
    # Tracking it over epochs reveals whether the model grows more confident
    # gradually or collapses into aggressive self-labeling too early.
    out_path = Path(output_path)

    rows = [
        row for row in history_rows
        if row.get("stage") == "finetune" and row.get("pseudo_label_ratio") is not None
    ]

    if not rows:
        return out_path

    plt.figure(figsize=(8, 4))
    sns.lineplot(x=[row["epoch"] for row in rows], y=[row["pseudo_label_ratio"] for row in rows])
    plt.xlabel("Epoch")
    plt.ylabel("Pseudo-label ratio")
    plt.title("Pseudo-label acceptance over epochs")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def project_embeddings(embeddings: np.ndarray, method: str = "umap", random_state: int = 42) -> np.ndarray:
    """Project high-dimensional embeddings into two dimensions for visualization."""
    projection_method = method.lower()
    array = np.asarray(embeddings)

    if projection_method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        return reducer.fit_transform(array)

    if projection_method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
        return reducer.fit_transform(array)

    reducer = PCA(n_components=2)
    return reducer.fit_transform(array)


def plot_embedding_projection(
    embeddings: np.ndarray, labels: Sequence[int],
    output_path: str | Path, label_names: Sequence[str] | None = None,
    method: str = "umap", random_state: int = 42) -> Path:
    """Plot a two-dimensional embedding projection colored by class label."""
    out_path = Path(output_path)

    projected = project_embeddings(np.asarray(embeddings), method=method, random_state=random_state)
    labels_arr = np.asarray(labels)

    legend_labels = [
        label_names[int(label)]
        if label_names is not None and int(label) < len(label_names)
        else str(label) for label in labels_arr
    ]

    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        x=projected[:, 0], y=projected[:, 1],
        hue=legend_labels, s=25, alpha=0.8, palette="tab10")
    plt.title(f"{method.upper()} projection of document embeddings")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_seed_metric_trend(all_runs: pl.DataFrame, output_path: str | Path, metric: str = "macro_f1") -> Path:
    """Plot a metric comparison across datasets, methods, and random seeds."""
    out_path = Path(output_path)

    if all_runs.is_empty() or metric not in all_runs.columns:
        return out_path

    frame = all_runs.to_pandas()
    plt.figure(figsize=(9, 5))
    sns.pointplot(data=frame, x="dataset", y=metric, hue="method", dodge=True, errorbar=None)
    plt.title(f"{metric.replace('_', ' ').title()} across seeds")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path
