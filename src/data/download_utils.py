"""
Utility helpers for exporting, normalising, and reading dataset bundles.

Fetching/labeling layer and the on-disk representation: 
schema normalisation, YAML config generation, and generic table I/O for several file formats.
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

from constants import BENCHMARK_DEFAULTS, BENCHMARK_RUN_PREFIX, DOCUMENT_COLUMNS, REQUIRED_COLUMNS


def save_frame(df: pl.DataFrame, path: Path) -> None:
    """
    Write a DataFrame using the format implied by the destination file suffix.

    .parquet => write_parquet; any other suffix => write_csv. 
    Parent directories are created automatically so the caller does not need to mkdir beforehand.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path) if path.suffix == ".parquet" else df.write_csv(path)


def empty_citations() -> pl.DataFrame:
    """Return an empty citation table with an explicit integer schema.

    Having an explicit dtype prevents Polars from defaulting to Utf8 on the
    empty frame, which would cause a schema mismatch when concatenating with
    real citation tables later.
    """
    return pl.DataFrame({
        "source": pl.Series("source", [], dtype=pl.Int64),
        "target": pl.Series("target", [], dtype=pl.Int64)
    })


def document_defaults() -> dict[str, Any]:
    """
    Columns that may be absent from a raw document table.

    Fill missing columns with safe neutral values rather than raising an error.
    """
    return {"title": "", "abstract": "", "venue": "", "publisher": "", "authors": "", "year": None}


def ensure_required_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise an arbitrary document table to the project schema.

    Steps
    -----
    1. Add a 'doc_id' column via row index when it is absent.
    2. Fill any missing REQUIRED_COLUMNS with their default values.
    3. Validate that a 'label' column is present (required for training).
    4. Reorder to DOCUMENT_COLUMNS prefix followed by any extra columns.

    After this function runs, downstream code can assume a stable column order
    with safe defaults for all required metadata fields.

    Raises ValueError when the 'label' column is missing.
    """
    out = df.with_row_index("doc_id") if "doc_id" not in df.columns else df
    defaults = document_defaults()

    for column in REQUIRED_COLUMNS:
        if column not in out.columns:
            out = out.with_columns(pl.lit(defaults[column]).alias(column))

    if "label" not in out.columns:
        raise ValueError("documents frame must include a 'label' column")

    # Preserve the canonical column order while keeping any user-defined extras.
    ordered = [col for col in DOCUMENT_COLUMNS if col in out.columns]
    extra   = [col for col in out.columns if col not in ordered]
    return out.select(ordered + extra)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary with a clear error when it is missing."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config template not found: {config_path}")
    return yaml.safe_load(config_path.read_text()) or {}


def save_benchmark_config(
    dataset_name: str, out_dir: Path, config_template: str | Path,
    run_prefix: str = BENCHMARK_RUN_PREFIX) -> None:
    """
    Generate the dataset-specific training config next to the exported CSV files.

    The function deep-merges dataset paths and split settings into the provided
    YAML template so the caller only needs to maintain one base config that
    gets patched for each new dataset.

    The run_prefix parameter defaults to BENCHMARK_RUN_PREFIX (constants.py).
    Override it to distinguish different model families in experiment logs.
    """
    config = load_yaml(config_template)
    config.setdefault("project", {})
    config.setdefault("data", {})

    config["project"]["benchmark"]  = dataset_name
    config["project"]["run_name"]   = f"{run_prefix}_{dataset_name}"
    config["project"]["output_dir"] = f"runs/{run_prefix}"
    config["project"]["cache_dir"]  = f"cache/{run_prefix}"

    # Merge dataset-specific paths and split defaults from BENCHMARK_DEFAULTS.
    config["data"].update({
        "documents":    str(out_dir / "documents.csv"),
        "citations":    str(out_dir / "citations.csv"),
        "baselines":    str(out_dir / "baselines.csv"),
        "label_column": "label", "source_col": "source", "target_col": "target",
        **BENCHMARK_DEFAULTS[dataset_name]
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def save_dataset_bundle(
    dataset_name: str, out_dir:  str | Path,
    documents: pl.DataFrame, config_template: str | Path, citations: pl.DataFrame | None) -> None:
    """
    Export normalised document and citation tables together with a matching config.

    One-stop function for writing a complete dataset bundle:
        - documents.csv (schema-normalised via ensure_required_columns)
        - citations.csv (empty placeholder when citations is None)
        - config.yaml   (generated from the template)
    """
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_frame(ensure_required_columns(documents), output_dir / "documents.csv")
    save_frame(citations if citations is not None else empty_citations(), output_dir / "citations.csv")
    save_benchmark_config(dataset_name, output_dir, config_template)


def mask_to_split(train_mask: Any, val_mask: Any, test_mask: Any) -> list[str]:
    """
    Convert boolean split masks into an explicit per-row split label column.

    Iterates the three masks in lock-step and assigns the label of whichever
    mask is True. Rows that are False in all three masks are labelled 'unassigned'; 
    this indicates a data preparation error and should be investigated.
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
    Recursively collect files matching any of the provided glob patterns.

    Uses a set internally to deduplicate paths that match multiple patterns,
    then returns a sorted list for deterministic ordering.
    """
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(found)


def frame_from_json_payload(payload: Any, path: Path) -> pl.DataFrame:
    """Convert a supported JSON payload shape into a Polars DataFrame.

    Supported shapes
    ----------------
    - A top-level JSON array:            [{...}, {...}, ...]
    - A dict with a well-known list key: {"rows": [...]} / {"data": [...]} /
                                         {"documents": [...]} / {"records": [...]}

    Raises ValueError for any other shape.
    """
    if isinstance(payload, list):
        return pl.DataFrame(payload)
    if isinstance(payload, dict):
        for key in ("rows", "data", "documents", "records"):
            if isinstance(value := payload.get(key), list):
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
