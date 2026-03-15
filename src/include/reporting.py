from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import polars as pl

from utils import PRIMARY_METRICS, to_frame


def aggregate_seed_results(
    rows: Sequence[Mapping[str, Any]], 
    group_cols: Sequence[str] = ("method", "dataset", "ablation"), 
    metrics: Sequence[str] = PRIMARY_METRICS) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Mean and standard deviation across multiple seeds for each configuration."""
    all_runs = to_frame(rows)
    if all_runs.is_empty():
        return all_runs, pl.DataFrame()

    # Aggregation
    # Push the  entire group-by, mean, and std computation down to its multi-threaded bakend, 
    # making this operation virtually instantaneous even for thousands of logged runs.
    valid_metrics = [m for m in metrics if m in all_runs.columns]
    agg_exprs = [pl.len().alias("num_runs")]
    
    for m in valid_metrics:
        # Prevent std-dev calculation errors on int columns
        col = pl.col(m).cast(pl.Float64)
        agg_exprs.extend([col.mean().alias(f"{m}_mean"), col.std(ddof=0).alias(f"{m}_std")])

    summary = all_runs.group_by(list(group_cols), maintain_order=True).agg(agg_exprs)
    return all_runs, summary


def format_score(value: float | None, std: float | None, precision: int = 4) -> str:
    """Safely formats a metric score alongside its standard deviation."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    if std is None or (isinstance(std, float) and not np.isfinite(std)):
        return f"{value:.{precision}f}"
    return f"{value:.{precision}f} $\\pm$ {std:.{precision}f}"


def decorate_ranked_cells(summary: pl.DataFrame, dataset: str, metric: str, method_col: str = "method") -> dict[str, str]:
    """Identifies the top two performing methods to format LaTeX cells with bolding/underlining."""
    # Decoupling Compute from Presentation
    # Numerical data so it can be passed to plotting functions later.
    # The presentation logic (LaTeX decoration) is strictly isolated to this rendering step.
    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
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
    metrics: Sequence[str] = PRIMARY_METRICS, method_col: str = "method", 
    drop_full_suffix: bool = True,) -> Path:
    """Generates and writes a publication-ready LaTeX table summarizing benchmark results."""
    #  End-to-End Academic Reproducibility
    # Making the pipeline generate raw LaTeX natively,  completely eliminate transcription errors.
    # When a new run finishes, the paper is updated instantly and accurately.
    path = Path(path)
    if summary.is_empty():
        path.write_text("% Empty benchmark table\n")
        return path

    datasets = list(datasets or summary["dataset"].unique().to_list())
    
    if drop_full_suffix and "ablation" in summary.columns:
        summary = summary.with_columns(
            pl.when(pl.col("ablation") == "full").then(pl.col(method_col))
            .otherwise(pl.col(method_col) + " [" + pl.col("ablation") + "]").alias(method_col))
        
    methods = summary[method_col].unique().to_list()

    # Construct LaTeX headers
    group_headers = " & " + " & ".join([
        f"\\multicolumn{{{len(metrics)}}}{{c}}{{{dataset}}}" for dataset in datasets
    ]) + r" \\"
    metrics_fmted = " & ".join(m.replace("_", "-").title() for _ in datasets for m in metrics)
    sub_headers = method_col.title() + " & " + metrics_fmted + r" \\"
    column_spec = "l" + "c" * (len(datasets) * len(metrics))
    
    rank_maps = {(dataset, metric): 
        decorate_ranked_cells(summary, dataset, metric, method_col) 
        for dataset in datasets for metric in metrics}

    # Construct LaTeX body rows
    body_lines = []
    for method in methods:
        row = [str(method)]
        for dataset in datasets:
            subset = summary.filter((pl.col(method_col) == method) & (pl.col("dataset") == dataset))
            values = subset.to_dicts()[0] if subset.height else {}
            
            for metric in metrics:
                cell = rank_maps[(dataset, metric)].get(
                    str(method), format_score(values.get(f"{metric}_mean"), values.get(f"{metric}_std"))
                )
                row.append(cell)
        body_lines.append(" & ".join(row) + r" \\")

    latex = "\n".join([
        r"\begin{tabular}{" + column_spec + "}", r"\toprule",
        group_headers, sub_headers,
        r"\midrule", *body_lines, r"\bottomrule", r"\end{tabular}"
    ])
    
    path.write_text(latex)
    return path
