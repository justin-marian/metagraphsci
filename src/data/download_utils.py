import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
import polars as pl
import yaml

from constants import BENCHMARK_DEFAULTS, DOCUMENT_COLUMNS, REQUIRED_COLUMNS


def save_frame(df: pl.DataFrame, path: Path) -> None:
    """Save a Polars DataFrame to disk, supporting both Parquet and CSV."""
    # Dual Format Support
    # CSV is kept for human-readability and legacy compatibility, but Parquet 
    # is supported (and preferred) because it preserves strict data types 
    # (LIKE Int64 vs Float64) and reads significantly faster for large graphs.
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path) if path.suffix == ".parquet" else df.write_csv(path)


def empty_citations() -> pl.DataFrame:
    """Generate an empty citation dataframe with strict typing."""
    # Strict Schema Fallbacks
    # If a dataset lacks a citation graph (e.g., text-only baselines), must return 
    # an empty frame rather than `None`. Explicitly defining `pl.Int64` ensures that 
    # downstream operations (like PyTorch tensor casting) don't crash trying to infer 
    # types from a 0-row table.
    return pl.DataFrame({
        "source": pl.Series(name="source", values=[], dtype=pl.Int64), 
        "target": pl.Series(name="target", values=[], dtype=pl.Int64)
    })


def ensure_required_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Fill missing columns and reorder fields to match the project schema."""
    # Defensive Data Normalization
    # External datasets are incredibly messy. Instead of littering the training loop 
    # with `if "abstract" in row:` checks, we force all incoming tables into a 
    # rigid, unified schema right at the ingestion boundary. Missing fields are 
    # filled with safe defaults so the Dataset/Model logic remains completely agnostic 
    # to the original data source's quirks.
    defaults: dict[str, Any] = {"title": "", "abstract": "", "venue": "", "publisher": "", "authors": "", "year": None}
    out = df.with_row_index("doc_id") if "doc_id" not in df.columns else df

    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out = out.with_columns(pl.lit(defaults[col]).alias(col))

    if "label" not in out.columns:
        raise ValueError("documents frame must include a 'label' column")

    # Reorder columns: required fields first, followed by any original extra metadata
    ordered = [c for c in DOCUMENT_COLUMNS if c in out.columns]
    extra = [c for c in out.columns if c not in ordered]
    return out.select(ordered + extra)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config template not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def save_benchmark_config(dataset_name: str, out_dir: Path, config_template: str | Path) -> None:
    """Write a benchmark-specific config next to the exported dataset files."""
    # Code-Config-Data Coupling
    # To guarantee reproducibility, the script that downloads and builds the data 
    # also generates the training config. This ensures that the dataset's specific 
    # requirements (like evaluation split strategies) are hardcoded into the yaml 
    # file that the trainer will eventually use, eliminating human configuration error.
    cfg = load_yaml(config_template)
    cfg.setdefault("project", {})
    cfg.setdefault("data", {})

    cfg["project"]["benchmark"] = dataset_name
    cfg["project"]["run_name"] = f"MetaGraphSci_{dataset_name}"
    cfg["data"].update({
        "documents": str(out_dir / "documents.csv"),
        "citations": str(out_dir / "citations.csv"),
        "baselines": str(out_dir / "baselines.csv")
    })

    for k, v in BENCHMARK_DEFAULTS[dataset_name].items():
        cfg["data"][k] = v

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def save_dataset_bundle(
    dataset_name: str, out_dir: str | Path, documents: pl.DataFrame,
    config_template: str | Path, citations: pl.DataFrame | None = None) -> None:
    """Export normalized tables and generate the config used by the pipeline."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    docs = ensure_required_columns(documents)
    cits = citations if citations is not None else empty_citations()

    save_frame(docs, out / "documents.csv")
    save_frame(cits, out / "citations.csv")
    save_benchmark_config(dataset_name, out, config_template)


def mask_to_split(train_mask, val_mask, test_mask) -> list[str]:
    """Convert boolean split masks into human-readable split names."""
    # Tabular Split Tracking over Graph Masks
    # PyTorch Geometric natively uses parallel boolean tensors (train_mask, val_mask) 
    # mapped to node indices. This is brittle if the graph nodes get reshuffled. 
    # By converting these masks into an explicit string column ("train", "val", "test") 
    # attached to the document dataframe, we can safely filter, sort, and sample the 
    # tabular data without losing the evaluation boundaries.
    split: list[str] = []
    for is_t, is_v, is_te in zip(train_mask.tolist(), val_mask.tolist(), test_mask.tolist()):
        if is_t: 
            split.append("train")
        elif is_v:
            split.append("val")
        elif is_te: 
            split.append("test")
        else: 
            split.append("unassigned")
    return split


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, path.open("wb") as file_handle:
        shutil.copyfileobj(response, file_handle)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(out_dir)


def find_candidates(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    """Find candidate files matching a set of glob patterns."""
    found: list[Path] = []
    for p in patterns:
        found.extend(root.rglob(p))
    return sorted({p for p in found if p.is_file()})


def read_table(path: Path) -> pl.DataFrame:
    """Robust I/O reader supporting multiple tabular and JSON layouts."""
    # Flexible Ingestion
    # Academic datasets are distributed in wildly inconsistent formats. 
    # This unified reader attempts to parse standard tables (CSV/Parquet) and 
    # heuristically unpacks nested JSON structures (extracting lists of records 
    # hidden behind "data" or "rows" keys) so the pipeline doesn't crash on new formats.
    if path.suffix == ".csv": 
        return pl.read_csv(path)
    if path.suffix == ".parquet": 
        return pl.read_parquet(path)
    if path.suffix == ".jsonl": 
        return pl.read_ndjson(path)
    
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list): 
            return pl.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("rows", "data", "documents", "records"):
                if key in payload and isinstance(payload[key], list):
                    return pl.DataFrame(payload[key])
        raise ValueError(f"Unsupported JSON table structure: {path}")
    raise ValueError(f"Unsupported table format: {path.suffix}")
