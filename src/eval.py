from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import umap
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, log_loss, matthews_corrcoef, precision_recall_fscore_support,
    precision_score, recall_score, roc_auc_score)

"""Evaluate runs and export analysis artifacts for MetaGraphSci.

This file turns raw predictions, probabilities, and embeddings into readable
reports. It keeps evaluation separate from training so metrics, plots, and result
tables can be reused across experiments without touching the trainer.

- compute the headline metrics used for comparison,
- build row-level prediction tables for error analysis,
- generate plots for confusion, calibration, history, and embeddings,
- aggregate multiple seeds into benchmark-friendly summaries.
"""


sns.set_theme(style="whitegrid")
PRIMARY_METRICS = ["accuracy", "micro_f1", "macro_f1", "balanced_accuracy", "mcc"]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_frame(rows: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    if isinstance(rows, Mapping):
        return pl.DataFrame([dict(rows)])
    return pl.DataFrame([dict(row) for row in rows])


def save_frame(frame: pl.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)), "cohen_kappa": float(cohen_kappa_score(y_true, y_pred))}
    if y_prob is not None:
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=np.arange(y_prob.shape[1])))
        try:
            metrics["macro_ovr_auroc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            metrics["macro_ovr_auroc"] = float("nan")
    return metrics


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: Sequence[str] | None = None) -> pl.DataFrame:
    """Build a class-by-class report for precision, recall, F1, and support."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)

    precision_arr = np.asarray(precision)
    recall_arr = np.asarray(recall)
    f1_arr = np.asarray(f1)
    support_arr = np.asarray(support)

    rows = []
    for idx, cls in enumerate(classes):
        class_id = int(cls)
        rows.append({
            "class_id": class_id, "class_name": (label_names[class_id] if label_names and class_id < len(label_names) else str(class_id)),
            "precision": float(precision_arr[idx]), "recall": float(recall_arr[idx]), "f1": float(f1_arr[idx]), "support": int(support_arr[idx])
        })
    return to_frame(rows)


def prediction_table(doc_ids: Sequence[int], y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> pl.DataFrame:
    """Create a per-document table of predictions, confidence, and correctness."""
    confidence = y_prob.max(axis=1) if y_prob is not None else np.full(len(y_true), np.nan)
    rows = [{
        "doc_id": int(doc_id), "label": int(true),"prediction": int(pred),
        "confidence": float(conf), "correct": int(true == pred)
    } for doc_id, true, pred, conf in zip(doc_ids, y_true, y_pred, confidence)]
    return to_frame(rows)


def evaluate_predictions(
    y_true: Sequence[int], y_pred: Sequence[int], y_prob: np.ndarray | None = None,
    doc_ids: Sequence[int] | None = None, label_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Bundle scalar metrics, class reports, and prediction rows in one structure."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = None if y_prob is None else np.asarray(y_prob)
    doc_id_arr = np.arange(len(y_true_arr)) if doc_ids is None else np.asarray(doc_ids)
    return {"metrics": multiclass_metrics(y_true_arr, y_pred_arr, y_prob_arr),
            "per_class": per_class_metrics(y_true_arr, y_pred_arr, label_names),
            "predictions": prediction_table(doc_id_arr, y_true_arr, y_pred_arr, y_prob_arr)}


def plot_confusion(
    y_true: Sequence[int],  y_pred: Sequence[int],
    label_names: Sequence[str] | None, output_path: str | Path,
    normalize: bool = True) -> Path:
    output_path = Path(output_path)
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    ticks = list(label_names) if label_names else [str(i) for i in range(cm.shape[0])]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
        xticklabels=ticks, yticklabels=ticks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Normalized confusion matrix" if normalize else "Confusion matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_training_history(
    history_rows: Sequence[Mapping[str, Any]],
    output_path: str | Path, metrics: Sequence[str] | None = None) -> Path:
    output_path = Path(output_path)
    metrics = list(metrics or ["train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_balanced_accuracy"])
    available = [metric for metric in metrics if any(metric in row and row[metric] is not None for row in history_rows)]
    if not available:
        raise ValueError("No requested metrics found in history_rows.")

    plt.figure(figsize=(10, 6))
    for metric in available:
        xs = [row["epoch"] for row in history_rows if row.get(metric) is not None]
        ys = [row[metric] for row in history_rows if row.get(metric) is not None]
        sns.lineplot(x=xs, y=ys, label=metric)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training and evaluation history")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_class_support(per_class: pl.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    frame = per_class.to_pandas().sort_values("support", ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=frame, x="class_name", y="support")
    plt.xticks(rotation=45, ha="right")
    plt.title("Class support")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_per_class_f1(per_class: pl.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    frame = per_class.to_pandas().sort_values("f1", ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=frame, x="class_name", y="f1")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Per-class F1")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_confidence_histogram(predictions: pl.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    frame = predictions.to_pandas()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=frame, x="confidence", hue="correct", stat="density", common_norm=False, bins=20, element="step")
    plt.title("Confidence by correctness")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_calibration(
    y_true: Sequence[int], y_prob: np.ndarray,
    output_path: str | Path, n_bins: int = 10) -> Path:
    output_path = Path(output_path)
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    pred = y_prob_arr.argmax(axis=1)
    conf = y_prob_arr.max(axis=1)
    correct = (pred == y_true_arr).astype(int)
    frac_pos, mean_pred = calibration_curve(correct, conf, n_bins=n_bins, strategy="uniform")

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title("Reliability diagram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_pseudo_label_ratio(history_rows: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    rows = [row for row in history_rows if row.get("stage") == "finetune" and row.get("pseudo_label_ratio") is not None]
    if not rows:
        return output_path

    plt.figure(figsize=(8, 4))
    sns.lineplot(x=[row["epoch"] for row in rows], y=[row["pseudo_label_ratio"] for row in rows])
    plt.xlabel("Epoch")
    plt.ylabel("Pseudo-label ratio")
    plt.title("Pseudo-label acceptance over epochs")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def project_embeddings(embeddings: np.ndarray, method: str = "umap", random_state: int = 42) -> np.ndarray:
    method = method.lower()
    if method == "umap":
        return umap.UMAP(n_components=2, random_state=random_state).fit_transform(embeddings)
    if method == "tsne":
        return TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto").fit_transform(embeddings)
    return PCA(n_components=2, random_state=random_state).fit_transform(embeddings)


def plot_embedding_projection(
    embeddings: np.ndarray, labels: Sequence[int],
    output_path: str | Path, label_names: Sequence[str] | None = None,
    method: str = "umap", random_state: int = 42) -> Path:
    """Plot a 2D embedding projection colored by label."""
    output_path = Path(output_path)
    projected = project_embeddings(np.asarray(embeddings), method=method, random_state=random_state)
    labels_arr = np.asarray(labels)
    legend = [label_names[int(label)] if label_names and int(label) < len(label_names) else str(label) for label in labels_arr]

    plt.figure(figsize=(9, 7))
    sns.scatterplot(x=projected[:, 0], y=projected[:, 1], hue=legend, s=25, alpha=0.8, palette="tab10")
    plt.title(f"{method.upper()} projection of document embeddings")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def aggregate_seed_results(
    rows: Sequence[Mapping[str, Any]],
    group_cols: Sequence[str] = ("method", "dataset", "ablation"),
    metrics: Sequence[str] = PRIMARY_METRICS) -> tuple[pl.DataFrame, pl.DataFrame]:
    all_runs = to_frame(rows)
    summary_rows: list[dict[str, Any]] = []
    if all_runs.is_empty():
        return all_runs, pl.DataFrame()

    groups = all_runs.partition_by(list(group_cols), maintain_order=True)
    for frame in groups:
        record = {col: frame[col][0] for col in group_cols if col in frame.columns}
        record["num_runs"] = frame.height
        for metric in metrics:
            if metric not in frame.columns:
                continue
            values = frame[metric].to_numpy().astype(float)
            record[f"{metric}_mean"] = float(np.nanmean(values))
            record[f"{metric}_std"] = float(np.nanstd(values, ddof=0))
        summary_rows.append(record)

    return all_runs, to_frame(summary_rows)


def format_score(value: float | None, std: float | None, precision: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    if std is None or (isinstance(std, float) and not np.isfinite(std)):
        return f"{value:.{precision}f}"
    return f"{value:.{precision}f} $\\pm$ {std:.{precision}f}"


def decorate_ranked_cells(summary: pl.DataFrame, dataset: str, metric: str, method_col: str = "method") -> dict[str, str]:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary.columns:
        return {}

    subset = summary.filter(pl.col("dataset") == dataset).sort(mean_col, descending=True, nulls_last=True)
    decorated: dict[str, str] = {}
    for rank, row in enumerate(subset.to_dicts(), start=1):
        cell = format_score(row.get(mean_col), row.get(std_col))
        if rank == 1:
            cell = f"\\textbf{{{cell}}}"
        elif rank == 2:
            cell = f"\\underline{{{cell}}}"
        decorated[str(row[method_col])] = cell
    return decorated


def save_benchmark_table(
    summary: pl.DataFrame, path: str | Path, datasets: Sequence[str] | None = None,
    metrics: Sequence[str] = PRIMARY_METRICS, method_col: str = "method", drop_full_suffix: bool = True) -> Path:
    path = Path(path)
    if summary.is_empty():
        path.write_text("% Empty benchmark table\n")
        return path

    datasets = list(datasets or summary["dataset"].unique().to_list())
    if drop_full_suffix and "ablation" in summary.columns:
        summary = summary.with_columns(
            pl.when(pl.col("ablation") == "full")
            .then(pl.col(method_col))
            .otherwise(pl.col(method_col) + " [" + pl.col("ablation") + "]")
            .alias(method_col))
    methods = summary[method_col].unique().to_list()

    group_headers = " & " + " & ".join([f"\\multicolumn{{{len(metrics)}}}{{c}}{{{dataset}}}" for dataset in datasets]) + r" \\" 
    sub_headers = method_col.title() + " & " + " & ".join(metric.replace("_", "-").title() for _ in datasets for metric in metrics) + r" \\" 
    column_spec = "l" + "c" * (len(datasets) * len(metrics))
    rank_maps = {(dataset, metric): decorate_ranked_cells(summary, dataset, metric, method_col=method_col) for dataset in datasets for metric in metrics}

    body_lines = []
    for method in methods:
        row = [str(method)]
        for dataset in datasets:
            subset = summary.filter((pl.col(method_col) == method) & (pl.col("dataset") == dataset))
            values = subset.to_dicts()[0] if subset.height else {}
            for metric in metrics:
                row.append(
                    rank_maps[(dataset, metric)].get(str(method), format_score(values.get(f"{metric}_mean"), values.get(f"{metric}_std"))))
        body_lines.append(" & ".join(row) + " \\")

    latex = "\n".join([
        r"\begin{tabular}{" + column_spec + "}", 
        r"\toprule", group_headers, sub_headers, 
        r"\midrule", *body_lines, r"\bottomrule", 
        r"\end{tabular}"])
    path.write_text(latex)
    return path


def plot_seed_metric_trend(all_runs: pl.DataFrame, output_path: str | Path, metric: str = "macro_f1") -> Path:
    """Plot one metric across seeds, datasets, and methods."""
    output_path = Path(output_path)
    if all_runs.is_empty() or metric not in all_runs.columns:
        return output_path

    frame = all_runs.to_pandas()
    plt.figure(figsize=(9, 5))
    sns.pointplot(data=frame, x="dataset", y=metric, hue="method", dodge=True, errorbar=None)
    plt.title(f"{metric.replace('_', ' ').title()} across seeds")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_evaluation_bundle(
    bundle: Mapping[str, Any], output_dir: str | Path, split: str,
    y_true: Sequence[int], y_pred: Sequence[int], y_prob: np.ndarray | None = None,
    embeddings: np.ndarray | None = None, label_names: Sequence[str] | None = None,
    history_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Path]:
    """Save the tables and plots produced for one evaluation split."""
    output_dir = ensure_dir(output_dir)
    paths = {
        "metrics_json": output_dir / f"{split}_metrics.json",  "per_class_csv": output_dir / f"{split}_per_class.csv",
        "predictions_csv": output_dir / f"{split}_predictions.csv", "confusion_png": output_dir / f"{split}_confusion.png",
        "class_support_png": output_dir / f"{split}_class_support.png", 
        "per_class_f1_png": output_dir / f"{split}_per_class_f1.png",
        "confidence_png": output_dir / f"{split}_confidence.png"}

    paths["metrics_json"].write_text(json.dumps(bundle["metrics"], indent=2))
    save_frame(bundle["per_class"], paths["per_class_csv"])
    save_frame(bundle["predictions"], paths["predictions_csv"])
    plot_confusion(y_true, y_pred, label_names, paths["confusion_png"], normalize=True)
    plot_class_support(bundle["per_class"], paths["class_support_png"])
    plot_per_class_f1(bundle["per_class"], paths["per_class_f1_png"])
    plot_confidence_histogram(bundle["predictions"], paths["confidence_png"])

    if y_prob is not None:
        paths["calibration_png"] = output_dir / f"{split}_calibration.png"
        plot_calibration(y_true, y_prob, paths["calibration_png"])

    if embeddings is not None:
        paths["embedding_png"] = output_dir / f"{split}_umap.png"
        plot_embedding_projection(embeddings, y_true, paths["embedding_png"], label_names=label_names, method="umap")

    if history_rows:
        paths["history_png"] = output_dir / "training_curves.png"
        plot_training_history(history_rows, paths["history_png"])
        paths["pseudo_ratio_png"] = output_dir / "pseudo_label_ratio.png"
        plot_pseudo_label_ratio(history_rows, paths["pseudo_ratio_png"])

    return paths
