"""
Shared cache-location helpers used by every disk caching module.

Centralising these utilities avoids drifting layouts across the four cache
modules (encoders, graph splits, tokenization, doc embeddings) and gives a
single place to compute deterministic fingerprints used as invalidation keys.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import polars as pl


def cache_root(project_cfg: dict[str, Any], seed: int | None = None) -> Path:
    """
    Resolve the on-disk cache directory for a benchmark.

    Seed-independent artefacts (tokenized corpus, doc embeddings) live under
    `<cache_dir>/global` so they are reused across seeds.  Seed-dependent
    artefacts (encoders, graph splits, neighbor caches) live under
    `<cache_dir>/seed_<n>` to mirror the existing context-cache layout.
    """
    base = Path(project_cfg["cache_dir"])
    scope = "global" if seed is None else f"seed_{int(seed)}"
    path = base / scope
    path.mkdir(parents=True, exist_ok=True)
    return path


def docs_fingerprint(documents: pl.DataFrame) -> str:
    """
    Deterministic content hash of the document table.

    Uses a stable digest over (row count, sorted doc_id list, column schema)
    so any structural change in the corpus invalidates dependent caches
    without requiring a full row-by-row hash of large frames.
    """
    doc_ids = sorted(int(v) for v in documents["doc_id"].to_list())
    parts = [
        str(documents.height),
        ",".join(str(c) for c in sorted(documents.columns)),
        ",".join(str(d) for d in doc_ids),
    ]
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def caching_enabled(cfg: dict[str, Any], artifact: str) -> bool:
    """
    Read the per-artefact caching toggle from config, defaulting to True.

    The `caching:` block is optional so existing configs keep working without
    modification; new configs can disable any individual cache by setting its
    flag to False.
    """
    return bool(cfg.get("caching", {}).get(artifact, True))
