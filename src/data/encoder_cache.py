"""
Disk caching for venue, publisher, and author encoders.

Encoders are deterministic functions of the training-split document table, so a
JSON cache keyed on ``(seed, train-doc fingerprint)`` avoids the O(N) Polars
unique/sort work on every run. The cache is human-readable so vocabulary shifts
across runs can be diffed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeAlias

import polars as pl
from loguru import logger

from .cache_utils import docs_fingerprint, metadata_matches
from .tabular_utils import create_encoders

Encoders: TypeAlias = dict[str, dict[str, int]]

COMPATIBILITY_KEYS = ("seed", "num_train_docs", "train_docs_fingerprint")


def compute_encoder_metadata(train_docs: pl.DataFrame, seed: int) -> dict[str, Any]:
    """Build a content-aware fingerprint for encoder vocabulary inputs."""
    return {
        "seed": int(seed),
        "num_train_docs": int(train_docs.height),
        "train_docs_fingerprint": docs_fingerprint(train_docs)}


def encoder_is_compatible(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return True when a saved encoder cache matches the current training split."""
    return metadata_matches("encoder", metadata, expected, COMPATIBILITY_KEYS)


def save_encoder_cache(encoders: Encoders, path: str | Path, metadata: dict[str, Any]) -> None:
    """Persist encoder vocabularies as JSON alongside their build metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"metadata": metadata, "encoders": encoders}
    output_path.write_text(json.dumps(payload, indent=2))

    sizes = {name: len(mapping) for name, mapping in encoders.items()}
    logger.info("Saved encoder cache: {} -> {}", sizes, output_path)


def load_encoder_cache(path: str | Path) -> tuple[Encoders, dict[str, Any]]:
    """Load encoder vocabularies and their build metadata from disk."""
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Encoder cache not found: {cache_path}")

    payload = json.loads(cache_path.read_text())
    encoders: Encoders = {
        name: {str(key): int(value) for key, value in mapping.items()}
        for name, mapping in payload["encoders"].items()}
    return encoders, payload["metadata"]


def build_encoder_cache(train_docs: pl.DataFrame) -> Encoders:
    """Build encoder vocabularies from the training-split documents."""
    venue, publisher, author = create_encoders(train_docs)
    return {"venue": venue, "publisher": publisher, "author": author}
