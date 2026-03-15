from pathlib import Path
from typing import Any, Mapping, Sequence
import polars as pl

PRIMARY_METRICS = ["accuracy", "micro_f1", "macro_f1", "balanced_accuracy", "mcc"]


def ensure_dir(path: str | Path) -> Path:
    """Ensures that a directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_frame(rows: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """Converts dictionaries or sequences of dictionaries into a Polars DataFrame."""
    if isinstance(rows, Mapping):
        return pl.DataFrame([dict(rows)])
    return pl.DataFrame([dict(row) for row in rows])


def save_frame(frame: pl.DataFrame, path: str | Path) -> Path:
    """Saves a Polars DataFrame to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path
