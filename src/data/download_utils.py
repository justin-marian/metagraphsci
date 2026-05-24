"""
Utility helpers for exporting, normalising, and reading dataset bundles.

This module connects the fetching/labeling layer with the on-disk representation:
schema normalisation, YAML config generation, and generic table I/O for several
file formats.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from .constants import (
    BENCHMARK_DEFAULTS_DATASETS, BENCHMARK_RUN_PREFIX,
    DOCUMENT_COLUMNS, REQUIRED_COLUMNS)


def save_frame(df: pl.DataFrame, path: Path) -> None:
    """
    Write a DataFrame using the format implied by the destination suffix.

    ``.parquet`` writes Parquet; any other suffix writes CSV. Parent directories
    are created automatically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.write_parquet(path)
        return
    df.write_csv(path)


def empty_citations() -> pl.DataFrame:
    """
    Return an empty citation table with an explicit integer schema.

    Explicit dtypes prevent Polars from defaulting to Utf8 on the empty frame,
    which would later mismatch real citation tables during concatenation.
    """
    return pl.DataFrame({
        "source": pl.Series("source", [], dtype=pl.Int64),
        "target": pl.Series("target", [], dtype=pl.Int64)})


def document_defaults() -> dict[str, Any]:
    """Return safe neutral defaults for optional document columns."""
    return {
        "title": "", "abstract": "", "venue": "",
        "publisher": "", "authors": "", "year": None}


def ensure_required_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalise an arbitrary document table to the project schema.

    Steps:
    1. Add ``doc_id`` via row index when absent.
    2. Fill missing REQUIRED_COLUMNS with neutral defaults.
    3. Require ``label`` for training.
    4. Reorder to DOCUMENT_COLUMNS prefix followed by user-defined extras.
    """
    out = df.with_row_index("doc_id") if "doc_id" not in df.columns else df
    defaults = document_defaults()

    for column in REQUIRED_COLUMNS:
        if column not in out.columns:
            out = out.with_columns(pl.lit(defaults[column]).alias(column))

    if "label" not in out.columns:
        raise ValueError("documents frame must include a 'label' column")

    # Keep canonical columns first while preserving any user-defined extras.
    ordered = [column for column in DOCUMENT_COLUMNS if column in out.columns]
    extra = [column for column in out.columns if column not in ordered]
    return out.select(ordered + extra)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary with a clear missing-file error."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config template not found: {config_path}")
    return yaml.safe_load(config_path.read_text()) or {}


def save_benchmark_config(
    dataset_name: str, out_dir: Path, config_template: str | Path,
    run_prefix: str = BENCHMARK_RUN_PREFIX) -> None:
    """
    Generate the dataset-specific training config next to exported CSV files.

    Dataset paths and split settings are merged into a base YAML template so the
    caller only maintains one base config across benchmark datasets.
    """
    config = load_yaml(config_template)
    config.setdefault("project", {})
    config.setdefault("data", {})

    config["project"].update({
        "benchmark": dataset_name, "run_name": f"{run_prefix}_{dataset_name}",
        "output_dir": f"runs/{run_prefix}", "cache_dir": f"cache/{run_prefix}"})

    # Dataset-specific paths plus split defaults from BENCHMARK_DEFAULTS_DATASETS.
    config["data"].update({
        "documents": str(out_dir / "documents.csv"),
        "citations": str(out_dir / "citations.csv"),
        "baselines": str(out_dir / "baselines.csv"),
        "label_column": "label", "source_col": "source", "target_col": "target",
        **BENCHMARK_DEFAULTS_DATASETS[dataset_name]})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def save_dataset_bundle(
    dataset_name: str, out_dir: str | Path, documents: pl.DataFrame,
    config_template: str | Path, citations: pl.DataFrame | None) -> None:
    """
    Export normalised document/citation tables and a matching benchmark config.

    Writes:
    - documents.csv, schema-normalised via ensure_required_columns
    - citations.csv, or an empty placeholder when citations is None
    - config.yaml, generated from the template
    """
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_frame(ensure_required_columns(documents), output_dir / "documents.csv")
    save_frame(citations if citations is not None else empty_citations(), output_dir / "citations.csv")
    save_benchmark_config(dataset_name, output_dir, config_template)


def mask_to_split(train_mask: Any, val_mask: Any, test_mask: Any) -> list[str]:
    """
    Convert boolean split masks into explicit per-row split labels.

    Rows false in all three masks are marked ``unassigned`` because that usually
    indicates a data preparation issue.
    """
    splits: list[str] = []

    for is_train, is_val, is_test in zip(train_mask.tolist(), val_mask.tolist(), test_mask.tolist()):
        if is_train:
            splits.append("train")
        elif is_val:
            splits.append("val")
        elif is_test:
            splits.append("test")
        else:
            splits.append("unassigned")

    return splits


def download_file(url: str, path: Path) -> None:
    """Download a remote file to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip archive into the target directory."""
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(out_dir)


def find_candidates(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    """
    Recursively collect files matching any provided glob pattern.

    A set deduplicates files matching multiple patterns; sorted output keeps
    candidate ordering deterministic.
    """
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(found)


def frame_from_json_payload(payload: Any, path: Path) -> pl.DataFrame:
    """
    Convert a supported JSON payload shape into a Polars DataFrame.

    Supported shapes:
    - top-level array: ``[{...}, {...}]``
    - dict with one of: ``rows``, ``data``, ``documents``, ``records``
    """
    if isinstance(payload, list):
        return pl.DataFrame(payload)

    if isinstance(payload, dict):
        for key in ("rows", "data", "documents", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return pl.DataFrame(value)

    raise ValueError(f"Unsupported JSON table structure in: {path}")


def read_table(path: Path) -> pl.DataFrame:
    """Read a supported tabular file into a Polars DataFrame."""
    if path.suffix == ".csv":
        return pl.read_csv(path)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix == ".jsonl":
        return pl.read_ndjson(path)
    if path.suffix == ".json":
        return frame_from_json_payload(json.loads(path.read_text()), path)

    raise ValueError(f"Unsupported table format: {path.suffix!r}")
