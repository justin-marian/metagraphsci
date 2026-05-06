"""
Tabular document loading, normalisation, and split utilities.

Converts raw CSV / Parquet document tables into the canonical
schema expected by the rest of the pipeline, builds vocabulary encoders for
categorical metadata fields, and creates train/val/test splits with either
time-based or stratified-random strategies.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from .constants import YEAR, REQUIRED_COLUMNS, UNKNOWN_TOKEN


def parse_authors(value: Any) -> list[str]:
    """
    Normalise heterogeneous author encodings into one clean list of strings.

    Input may arrive in several forms depending on the data source:
    - A Python list (already parsed by Polars from a list column).
    - A Python-repr string like "['Alice', 'Bob']" (common in CSV exports).
    - A semicolon- or pipe-delimited string like "Alice;Bob" or "Alice|Bob".
    - None or float NaN (missing metadata).

    All forms are reduced to a list of non-empty stripped strings.  The order
    of the original author list is preserved.
    """
    if isinstance(value, list):
        # Already a list just clean each element.
        return [str(item).strip() for item in value if str(item).strip()]

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    # Try to parse Python-repr lists produced by CSV round-trips.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]

    # Fall back to semicolon / pipe delimiter (common in bibliography exports).
    return [part.strip() for part in text.replace("|", ";").split(";") if part.strip()]


def read_documents_frame(path: Path) -> pl.DataFrame:
    """Read a document table from a supported file format into a Polars DataFrame."""
    if path.suffix == ".csv":
        return pl.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path.suffix!r}")


def label_names_from_numeric(df: pl.DataFrame, label_column: str) -> list[str]:
    """
    Return sorted string label names for an already-numeric label column.

    The sorted order must exactly match the integer encoding assumed elsewhere
    (label 0 => first sorted name, label 1 => second, etc.) so model outputs
    can be decoded without ambiguity.
    """
    values = df.drop_nulls(label_column)[label_column].unique().to_list()
    return [str(label) for label in sorted(values)]




def map_numeric_labels(df: pl.DataFrame, label_column: str) -> tuple[pl.DataFrame, list[str]]:
    """Map arbitrary numeric labels to contiguous ids required by CrossEntropyLoss."""
    values = sorted(int(v) for v in df.drop_nulls(label_column)[label_column].unique().to_list())
    mapping = {old: new for new, old in enumerate(values)}
    mapped = df.with_columns(
        pl.col(label_column)
        .cast(pl.Int64, strict=False)
        .replace(mapping)
        .cast(pl.Int64)
        .alias("label")
    )
    return mapped, [str(v) for v in values]

def map_string_labels(df: pl.DataFrame, label_column: str) -> tuple[pl.DataFrame, list[str]]:
    """
    Map string labels to contiguous integer ids in deterministic sorted order.

    Alphabetical sorting is used so the mapping is stable across Python
    sessions and does not depend on dict insertion order or random state.
    A new 'label' column (int64) replaces the original string column.
    """
    classes = sorted(df.drop_nulls(label_column)[label_column].cast(pl.String).unique().to_list())
    mapping = {name: idx for idx, name in enumerate(classes)}
    mapped  = df.with_columns(pl.col(label_column).cast(pl.String).replace(mapping).cast(pl.Int64).alias("label"))
    return mapped, classes


def prepare_documents(documents: pl.DataFrame, label_column: str = "label") -> tuple[pl.DataFrame, list[str] | None]:
    """
    Normalise a raw document table into the strict pipeline schema.

    Steps performed
    ---------------
    1. Validate that all REQUIRED_COLUMNS are present.
    2. Cast each column to its canonical dtype and fill nulls with safe defaults.
    3. Normalise the author column via parse_authors (handles all input formats).
    4. Encode string labels to contiguous integers when a label_column exists.

    Returns the normalised DataFrame and, when labels are present, the list of
    class names indexed by their integer code.

    Raises ValueError if any required column is missing.
    """
    df = documents.clone()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.with_columns([
        pl.col("doc_id").cast(pl.Int64, strict=False),
        pl.col("title").fill_null("").cast(pl.String),
        pl.col("abstract").fill_null("").cast(pl.String),
        pl.col("venue").fill_null(UNKNOWN_TOKEN).cast(pl.String),
        pl.col("publisher").fill_null(UNKNOWN_TOKEN).cast(pl.String),
        pl.col("authors").map_elements(parse_authors, return_dtype=pl.List(pl.String)),
        pl.col("year").fill_null(YEAR).cast(pl.Int64, strict=False)
    ])

    label_names: list[str] | None = None
    if label_column in df.columns:
        if df[label_column].dtype.is_numeric():
            # External numeric labels are not guaranteed to be zero-based or contiguous.
            # CrossEntropyLoss expects targets in [0, num_classes).
            df, label_names = map_numeric_labels(df, label_column)
        else:
            # String labels: sort => assign contiguous integers => store mapping.
            df, label_names = map_string_labels(df, label_column)

    return df, label_names


def load_documents(path: str | Path, label_column: str = "label") -> tuple[pl.DataFrame, list[str] | None]:
    """
    Read a document file from disk and immediately normalise it.

    Convenience wrapper around read_documents_frame + prepare_documents for
    the common case of loading from a path rather than a pre-built DataFrame.
    """
    return prepare_documents(read_documents_frame(Path(path)), label_column=label_column)


def create_encoders(documents: pl.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """
    Build 1-indexed vocabularies for venue, publisher, and author fields.

    The UNKNOWN_TOKEN is always assigned index 0 so that:
        - Padding slots in author_ids tensors automatically map to 0.
        - Unseen entities at inference time map to 0 without a KeyError.

    Real entities start at index 1 and are sorted alphabetically to make the
    encoding deterministic across runs.
    """
    def encoder(values: list[str]) -> dict[str, int]:
        enc = {name: idx + 1 for idx, name in enumerate(sorted(values))}
        enc[UNKNOWN_TOKEN] = 0  # Index 0 is reserved for unknown / padding.
        return enc

    venue_values     = documents.drop_nulls("venue")["venue"].cast(pl.String).unique().to_list()
    publisher_values = documents.drop_nulls("publisher")["publisher"].cast(pl.String).unique().to_list()
    # Explode the list column before collecting unique author names.
    author_values    = documents.select(pl.col("authors").explode()).drop_nulls()["authors"].unique().to_list()
    return encoder(venue_values), encoder(publisher_values), encoder(author_values)


def split_documents(
    documents: pl.DataFrame,
    test_size: float, val_size: float, seed: int,
    strategy: str, time_column: str = "year"
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Create train, validation, and test splits.

    Strategies
    ----------
    "time" - Sort the dataset chronologically (then by doc_id for ties) and
        cut deterministic slices.  Train gets the oldest papers, test
        gets the most recent, reflecting the real-world deployment
        scenario where a model must generalise to future publications.
        Requires the time_column to be present and sortable.

    "random" - Stratified random split using sklearn.  Each class appears in
        roughly the same proportion across all three splits.  Preferred
        when temporal ordering is not meaningful for the task.

    In both cases, the frame must have a fully non-null 'label' column.  Rows
    with missing labels should be separated before calling this function (see
    create_low_label_split for the semi-supervised use case).

    Raises
    ------
    ValueError - If the 'label' column is absent or contains nulls, or if an
        unknown strategy is requested, or if any split is empty after the time-based cut.
    """
    if "label" not in documents.columns or documents["label"].is_null().any():
        raise ValueError("split_documents requires a fully labeled DataFrame.")

    strategy = strategy.lower()

    if strategy == "time":
        # Sort chronologically; doc_id breaks ties deterministically.
        ranked  = documents.sort([time_column, "doc_id"])
        total   = ranked.height
        n_test  = max(1, int(round(total * test_size)))
        n_val   = max(1, int(round(total * val_size)))
        n_train = max(0, total - n_test - n_val)

        # Oldest papers => train; middle band => val; most recent => test.
        train_df = ranked.slice(0, n_train)
        val_df   = ranked.slice(n_train, n_val)
        test_df  = ranked.slice(total - n_test, n_test)

        if train_df.is_empty() or val_df.is_empty() or test_df.is_empty():
            raise ValueError("Time-based split produced an empty partition.")

        return train_df, val_df, test_df

    if strategy != "random":
        raise ValueError(f"Unknown split strategy: {strategy!r}")

    # Stratified random split go through pandas because sklearn's
    # train_test_split works natively on DataFrame objects.
    pandas_docs = documents.to_pandas()
    train_pd, test_pd = train_test_split(pandas_docs, test_size=test_size, 
        random_state=seed, stratify=pandas_docs["label"])
    train_pd, val_pd = train_test_split(train_pd, test_size=val_size / (1.0 - test_size), 
        random_state=seed, stratify=train_pd["label"])
    return pl.from_pandas(train_pd), pl.from_pandas(val_pd), pl.from_pandas(test_pd)


def create_low_label_split(documents: pl.DataFrame, label_ratio: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split a labeled frame into labeled and unlabeled semi-supervised subsets.

    Within each class a random subset of size ceil(n * label_ratio) is kept
    labeled; the rest have their label set to null (unlabeled).
    Shuffling is applied after concat so the labeled and unlabeled sets are not ordered by
    class, which matters for DataLoaders without shuffle=True.

    The labeled/unlabeled split is stratified per class so every class has at
    least one labeled example regardless of label_ratio.
    """
    rng = np.random.default_rng(seed)
    labeled_parts:   list[pl.DataFrame] = []
    unlabeled_parts: list[pl.DataFrame] = []

    for label in sorted(documents.drop_nulls("label")["label"].cast(pl.Int64).unique().to_list()):
        group = documents.filter(pl.col("label") == label)
        order = rng.permutation(group.height)
        # Guarantee at least one labeled example per class.
        keep  = min(group.height, max(1, int(round(group.height * label_ratio))))
        labeled_parts.append(group[order[:keep]])
        unlabeled_parts.append(group[order[keep:]])

    labeled   = pl.concat(labeled_parts,   how="vertical").sample(fraction=1.0, seed=seed, shuffle=True)
    unlabeled = pl.concat(unlabeled_parts, how="vertical").sample(fraction=1.0, seed=seed, shuffle=True)

    # Null out the label column in the unlabeled portion so the dataset layer
    # can identify and skip these rows when computing the supervised loss.
    return labeled, unlabeled.with_columns(pl.lit(None).cast(pl.Int64).alias("label"))


def build_year_lookup(documents: pl.DataFrame) -> dict[int, int]:
    """
    Build a fast doc_id => year lookup from a prepared document frame.

    Used by the caching layer for temporal scoring without repeatedly querying
    the full DataFrame by row. Assumes the frame has been passed through preparation
    of documents in such wat that doc ids are unique and years are filled with safe defaults.
    """
    return dict(zip(documents["doc_id"].to_list(), documents["year"].to_list()))
