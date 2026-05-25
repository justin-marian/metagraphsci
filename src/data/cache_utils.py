"""
Shared helpers for the disk-caching layer.

The cache modules (tokenization, embedding, encoder, graph, neighbor) all share
the same low-level operations: hashing document tables, fingerprinting citation
edges, comparing saved metadata against an expected signature, and persisting
small JSON sidecars. Centralising those helpers here keeps each cache module
focused on its own artifact-specific build logic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

# Columns that affect cache validity. A change in any of these for any document
# must invalidate downstream tokens, embeddings, encoders, and labels.
DOC_HASH_COLUMNS = (
    "doc_id", "title", "abstract",
    "venue", "publisher", "authors", "year",
    "label")


def normalise_value(value: Any) -> Any:
    """Convert Polars/Python values into deterministic JSON-compatible values."""
    if value is None:
        return None

    if hasattr(value, "to_list"):
        return normalise_value(value.to_list())

    if isinstance(value, tuple | list):
        return [normalise_value(item) for item in value]

    if isinstance(value, dict):
        return {str(key): normalise_value(value[key]) for key in sorted(value)}

    try:
        if value != value:  # noqa: PLR0124 - portable NaN check
            return None
    except Exception:
        return value

    return value


def _doc_hash_payload(row: dict[str, Any]) -> str:
    """Serialise one document row's content-bearing fields for hashing."""
    payload = {col: normalise_value(row.get(col)) for col in DOC_HASH_COLUMNS}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# Process-local memo keyed on (id(df), height). Cache validation calls both
# per_doc_hashes and docs_fingerprint repeatedly across modules in one run, and
# the underlying O(N) row walk is the dominant cost on warm-cache runs.
_DOC_HASH_MEMO: dict[tuple[int, int], tuple[dict[int, str], str]] = {}


def doc_hashes_and_fingerprint(documents: pl.DataFrame) -> tuple[dict[int, str], str]:
    """Return per-document hashes and the aggregate corpus fingerprint."""
    key = (id(documents), int(documents.height))
    cached = _DOC_HASH_MEMO.get(key)
    if cached is not None:
        return cached

    hashes: dict[int, str] = {}
    for row in documents.iter_rows(named=True):
        hashes[int(row["doc_id"])] = hashlib.sha1(_doc_hash_payload(row).encode("utf-8")).hexdigest()

    payload = json.dumps(
        {str(doc_id): hashes[doc_id] for doc_id in sorted(hashes)},
        sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha1(payload.encode("utf-8")).hexdigest()

    # Bounded memo: document frames are large and short-lived.
    if len(_DOC_HASH_MEMO) > 8:
        _DOC_HASH_MEMO.clear()
    _DOC_HASH_MEMO[key] = (hashes, fingerprint)
    return hashes, fingerprint


def per_doc_hashes(documents: pl.DataFrame) -> dict[int, str]:
    """Return ``{doc_id: sha1}`` for the content-bearing fields of each document."""
    return doc_hashes_and_fingerprint(documents)[0]


def docs_fingerprint(documents: pl.DataFrame) -> str:
    """Return a deterministic content-aware fingerprint for the document table."""
    return doc_hashes_and_fingerprint(documents)[1]


def coerce_hash_dict(metadata: dict[str, Any], key: str = "per_doc_hashes") -> dict[int, str]:
    """Pull and normalise the per-doc hash dict embedded in a metadata sidecar."""
    return {int(k): str(v) for k, v in metadata.get(key, {}).items()}


def _edges_fingerprint_from_frame(citations: pl.DataFrame, source_col: str, target_col: str) -> str:
    """Hash sorted (source, target) pairs of an edge frame."""
    if citations.height == 0:
        return hashlib.sha1(b"[]").hexdigest()

    edge_frame = citations.select([source_col, target_col]).drop_nulls().sort([source_col, target_col])

    # Numeric columns have no JSON-escapable characters in their string form, so
    # the vectorised payload is byte-identical to json.dumps over (str, str) tuples.
    # For non-numeric (or mixed) columns we fall back to the original path.
    src_dtype = edge_frame.schema[source_col]
    dst_dtype = edge_frame.schema[target_col]
    if src_dtype.is_numeric() and dst_dtype.is_numeric():
        joined = edge_frame.with_columns(
            pl.concat_str([
                pl.lit('["'), pl.col(source_col).cast(pl.String),
                pl.lit('","'), pl.col(target_col).cast(pl.String),
                pl.lit('"]')]).alias("__edge"))
        payload = "[" + ",".join(joined["__edge"].to_list()) + "]"
    else:
        payload = json.dumps(
            [(str(src), str(dst)) for src, dst in edge_frame.iter_rows()],
            separators=(",", ":"))

    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# Process-local memo keyed on (path, mtime_ns, size, source_col, target_col).
# Repeated cache validations within one run avoid re-reading the citations CSV.
_PATH_FP_MEMO: dict[tuple[str, int, int, str, str], str] = {}


def _path_stat_key(path: str | Path, source_col: str, target_col: str) -> tuple[str, int, int, str, str] | None:
    """Build a memo key from a file's stat info, or None if the file is missing."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size), source_col, target_col)


def citation_edges_fingerprint(citations: pl.DataFrame | str | Path, source_col: str, target_col: str) -> str:
    """Return a deterministic fingerprint of sorted citation edges."""
    if isinstance(citations, pl.DataFrame):
        return _edges_fingerprint_from_frame(citations, source_col, target_col)

    key = _path_stat_key(citations, source_col, target_col)
    if key is not None and key in _PATH_FP_MEMO:
        return _PATH_FP_MEMO[key]

    digest = _edges_fingerprint_from_frame(pl.read_csv(citations), source_col, target_col)
    if key is not None:
        if len(_PATH_FP_MEMO) > 8:
            _PATH_FP_MEMO.clear()
        _PATH_FP_MEMO[key] = digest
    return digest


def stable_int_fingerprint(values: Any) -> str:
    """Hash a sorted integer collection for compact deterministic metadata."""
    payload = json.dumps(sorted(int(value) for value in values), separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def edge_set_fingerprint(edge_set: set[tuple[int, int]]) -> str:
    """Hash a citation edge set by streaming sorted pairs into SHA1."""
    if not edge_set:
        return hashlib.sha1(b"[]").hexdigest()

    # Stream directly into the SHA1 state to avoid materialising a giant
    # intermediate list-of-lists and JSON string. Byte-identical to
    # json.dumps([[s, d] for s, d in sorted(edge_set)], separators=(",", ":")).
    digest = hashlib.sha1()
    digest.update(b"[")
    first = True
    for src, dst in sorted(edge_set):
        if first:
            first = False
        else:
            digest.update(b",")
        digest.update(b"[")
        digest.update(str(int(src)).encode("ascii"))
        digest.update(b",")
        digest.update(str(int(dst)).encode("ascii"))
        digest.update(b"]")
    digest.update(b"]")
    return digest.hexdigest()


def cache_root(project_cfg: dict[str, Any], seed: int | None = None) -> Path:
    """Resolve and create the on-disk cache directory for one cache scope."""
    base = Path(project_cfg["cache_dir"])
    scope = "global" if seed is None else f"seed_{int(seed)}"
    path = base / scope
    path.mkdir(parents=True, exist_ok=True)
    return path


def global_cache_path(project_cfg: dict[str, Any], name: str) -> Path:
    """Return a seed-independent cache file path under ``global``."""
    return cache_root(project_cfg) / name


def seed_cache_path(project_cfg: dict[str, Any], seed: int, name: str) -> Path:
    """Return a seed-dependent cache file path under ``seed_<n>``."""
    return cache_root(project_cfg, seed=seed) / name


def caching_enabled(cfg: dict[str, Any], artifact: str) -> bool:
    """Read a per-artifact caching toggle from config, defaulting to True."""
    return bool(cfg.get("caching", {}).get(artifact, True))


def metadata_matches(name: str, metadata: dict[str, Any], expected: dict[str, Any], keys: tuple[str, ...]) -> bool:
    """
    Return True when every key in ``keys`` is present and equal in both dicts.

    Used by every cache module to compare a saved sidecar against the signature
    that would be produced for the current run. Logs which keys were missing or
    mismatched so cache-miss diagnostics are uniform across modules.
    """
    missing = [key for key in keys if key not in metadata]
    if missing:
        logger.info("{} cache incompatible: missing metadata keys {}", name, missing)
        return False

    mismatched = [key for key in keys if metadata.get(key) != expected.get(key)]
    if mismatched:
        logger.info("{} cache incompatible: mismatched metadata keys {}", name, mismatched)
        return False

    return True


def log_cache_hit(name: str, path: str | Path, reason: str = "compatible") -> None:
    """Emit a standard cache-hit log line."""
    logger.info("{} cache hit: {} ({})", name, Path(path), reason)


def log_cache_miss(name: str, path: str | Path, reason: str) -> None:
    """Emit a standard cache-miss log line."""
    logger.info("{} cache miss: {} ({})", name, Path(path), reason)


def log_cache_summary(name: str, path: str | Path, summary: dict[str, Any]) -> None:
    """Emit a standard cache summary with reuse/rebuild/remove counters."""
    logger.info("{} cache updated: {} {}", name, Path(path), summary)


def write_meta_sidecar(path: str | Path, metadata: dict[str, Any]) -> None:
    """Write the JSON sidecar that accompanies a torch-saved cache file."""
    # No indent: per_doc_hashes can be 50k+ entries, indented JSON is 5x slower
    # to serialise/parse and adds no value since this file is machine-read.
    Path(path).with_suffix(".meta.json").write_text(json.dumps(metadata))


def read_meta_sidecar(path: str | Path) -> dict[str, Any]:
    """Read the JSON sidecar that accompanies a torch-saved cache file."""
    return json.loads(Path(path).with_suffix(".meta.json").read_text())
