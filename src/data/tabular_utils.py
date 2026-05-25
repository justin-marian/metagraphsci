"""
Tabular document loading, normalisation, and split utilities.

Only active pipeline helpers are kept here: document parsing/preparation,
metadata encoder construction, train/val/test splitting, low-label splitting,
and the doc_id -> year lookup used by neighbour-cache scoring.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from .constants import REQUIRED_COLUMNS, UNKNOWN_TOKEN, YEAR

__all__ = [
    "parse_authors", "read_documents_frame", "map_numeric_labels",
    "map_string_labels", "prepare_documents", "load_documents",
    "create_encoders", "split_documents", "create_low_label_split",
    "build_year_lookup"]


def parse_authors(value: Any) -> list[str]:
    """Normalise heterogeneous author encodings into a clean list of names."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None

        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]

    return [part.strip() for part in text.replace("|", ";").split(";") if part.strip()]


def read_documents_frame(path: Path) -> pl.DataFrame:
    """Read a document table from CSV or Parquet."""
    if path.suffix == ".csv":
        return pl.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path.suffix!r}")


def map_numeric_labels(df: pl.DataFrame, label_column: str) -> tuple[pl.DataFrame, list[str]]:
    """Map arbitrary numeric labels to contiguous ids for CrossEntropyLoss."""
    values = sorted(int(value) for value in df.drop_nulls(label_column)[label_column].unique().to_list())
    mapping = {old: new for new, old in enumerate(values)}
    mapped = df.with_columns(
        pl.col(label_column).cast(pl.Int64, strict=False).replace(mapping).cast(pl.Int64).alias("label"))
    return mapped, [str(value) for value in values]


def map_string_labels(df: pl.DataFrame, label_column: str) -> tuple[pl.DataFrame, list[str]]:
    """Map string labels to deterministic contiguous integer ids."""
    classes = sorted(df.drop_nulls(label_column)[label_column].cast(pl.String).unique().to_list())
    mapping = {name: idx for idx, name in enumerate(classes)}
    mapped = df.with_columns(
        pl.col(label_column).cast(pl.String).replace(mapping).cast(pl.Int64).alias("label"))
    return mapped, classes


def prepare_documents(documents: pl.DataFrame, label_column: str = "label") -> tuple[pl.DataFrame, list[str] | None]:
    """Normalise a raw document table into the canonical pipeline schema."""
    df = documents.clone()
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.with_columns([
        pl.col("doc_id").cast(pl.Int64, strict=False),
        pl.col("title").fill_null("").cast(pl.String),
        pl.col("abstract").fill_null("").cast(pl.String),
        pl.col("venue").fill_null(UNKNOWN_TOKEN).cast(pl.String),
        pl.col("publisher").fill_null(UNKNOWN_TOKEN).cast(pl.String),
        pl.col("authors").map_elements(parse_authors, return_dtype=pl.List(pl.String)),
        pl.col("year").fill_null(YEAR).cast(pl.Int64, strict=False)])

    label_names: list[str] | None = None
    if label_column in df.columns:
        if df[label_column].dtype.is_numeric():
            df, label_names = map_numeric_labels(df, label_column)
        else:
            df, label_names = map_string_labels(df, label_column)

    return df, label_names


def load_documents(path: str | Path, label_column: str = "label") -> tuple[pl.DataFrame, list[str] | None]:
    """Read and normalise a document file."""
    return prepare_documents(read_documents_frame(Path(path)), label_column=label_column)


def create_encoders(documents: pl.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Build 1-indexed venue, publisher, and author vocabularies."""
    def encoder(values: list[str]) -> dict[str, int]:
        encoded = {name: idx + 1 for idx, name in enumerate(sorted(values))}
        encoded[UNKNOWN_TOKEN] = 0
        return encoded

    venue_values = documents.drop_nulls("venue")["venue"].cast(pl.String).unique().to_list()
    publisher_values = documents.drop_nulls("publisher")["publisher"].cast(pl.String).unique().to_list()
    author_values = (
        documents.select(pl.col("authors").explode())
        .drop_nulls()["authors"].unique().to_list())

    return encoder(venue_values), encoder(publisher_values), encoder(author_values)


def split_documents(
    documents: pl.DataFrame, test_size: float, val_size: float, seed: int, 
    strategy: str, time_column: str = "year") -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create train, validation, and test splits."""
    if "label" not in documents.columns or documents["label"].is_null().any():
        raise ValueError("split_documents requires a fully labeled DataFrame.")

    strategy = strategy.lower()

    if strategy == "time":
        ranked = documents.sort([time_column, "doc_id"])
        total = ranked.height
        n_test = max(1, int(round(total * test_size)))
        n_val = max(1, int(round(total * val_size)))
        n_train = max(0, total - n_test - n_val)

        train_df = ranked.slice(0, n_train)
        val_df = ranked.slice(n_train, n_val)
        test_df = ranked.slice(total - n_test, n_test)

        if train_df.is_empty() or val_df.is_empty() or test_df.is_empty():
            raise ValueError("Time-based split produced an empty partition.")

        return train_df, val_df, test_df

    if strategy != "random":
        raise ValueError(f"Unknown split strategy: {strategy!r}")

    pandas_docs = documents.to_pandas()

    # StratifiedShuffleSplit requires >= 2 samples per class. Force singletons into train.
    counts = pandas_docs["label"].value_counts()
    singleton_labels = set(counts[counts < 2].index)
    singletons = pandas_docs[pandas_docs["label"].isin(singleton_labels)]
    splittable = pandas_docs[~pandas_docs["label"].isin(singleton_labels)]

    train_pd, test_pd = train_test_split(
        splittable, test_size=test_size, random_state=seed,
        stratify=splittable["label"])
    train_pd, val_pd = train_test_split(
        train_pd, test_size=val_size / (1.0 - test_size),
        random_state=seed, stratify=train_pd["label"])

    if not singletons.empty:
        import pandas as pd
        train_pd = pd.concat([train_pd, singletons], ignore_index=True)

    return pl.from_pandas(train_pd), pl.from_pandas(val_pd), pl.from_pandas(test_pd)


def create_low_label_split(documents: pl.DataFrame, label_ratio: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split labeled data into labeled and unlabeled semi-supervised subsets."""
    rng = np.random.default_rng(seed)
    labeled_parts: list[pl.DataFrame] = []
    unlabeled_parts: list[pl.DataFrame] = []

    labels = sorted(documents.drop_nulls("label")["label"].cast(pl.Int64).unique().to_list())
    for label in labels:
        group = documents.filter(pl.col("label") == label)
        order = rng.permutation(group.height)
        keep = min(group.height, max(1, int(round(group.height * label_ratio))))

        labeled_parts.append(group[order[:keep]])
        if keep < group.height:
            unlabeled_parts.append(group[order[keep:]])

    labeled = pl.concat(labeled_parts, how="vertical").sample(fraction=1.0, seed=seed, shuffle=True)
    unlabeled = (
        pl.concat(unlabeled_parts, how="vertical").sample(fraction=1.0, seed=seed, shuffle=True)
        if unlabeled_parts else documents.head(0))

    return labeled, unlabeled.with_columns(pl.lit(None).cast(pl.Int64).alias("label"))


def build_year_lookup(documents: pl.DataFrame) -> dict[int, int]:
    """Build a fast doc_id -> year lookup from a prepared document frame."""
    return {
        int(doc_id): int(year)
        for doc_id, year in zip(documents["doc_id"].to_list(), documents["year"].to_list())}
