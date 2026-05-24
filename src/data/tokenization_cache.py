"""Incremental disk caching for pre-tokenised document text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import torch
from loguru import logger
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .cache_utils import (
    coerce_hash_dict, docs_fingerprint, log_cache_miss, log_cache_summary,
    metadata_matches, per_doc_hashes, read_meta_sidecar, write_meta_sidecar)

TokenizedLookup = dict[int, dict[str, Tensor]]

COMPATIBILITY_KEYS = ("tokenizer_name", "max_seq_length")


def compute_tokenization_metadata(documents: pl.DataFrame, tokenizer_name: str, max_seq_length: int) -> dict[str, Any]:
    """Fingerprint inputs that determine the tokenised tensor contents."""
    return {
        "tokenizer_name": str(tokenizer_name),
        "max_seq_length": int(max_seq_length),
        "num_docs": int(documents.height),
        "docs_fingerprint": docs_fingerprint(documents),
        "per_doc_hashes": per_doc_hashes(documents)}


def tokenization_is_compatible(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return True when tokeniser settings match and incremental hashes exist."""
    return (
        metadata_matches("tokenization", metadata, expected, COMPATIBILITY_KEYS)
        and isinstance(metadata.get("per_doc_hashes"), dict))


def save_tokenization_cache(lookup: TokenizedLookup, path: str | Path, metadata: dict[str, Any]) -> None:
    """Persist tokenised tensors as a single torch file plus a JSON sidecar."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc_ids = sorted(lookup.keys())
    seq_len = int(metadata["max_seq_length"])
    if doc_ids:
        input_ids = torch.stack([lookup[d]["input_ids"] for d in doc_ids]).cpu()
        attention_mask = torch.stack([lookup[d]["attention_mask"] for d in doc_ids]).cpu()
    else:
        input_ids = torch.empty((0, seq_len), dtype=torch.long)
        attention_mask = torch.empty((0, seq_len), dtype=torch.long)

    torch.save({
        "doc_ids": torch.tensor(doc_ids, dtype=torch.long),
        "input_ids": input_ids,
        "attention_mask": attention_mask}, output_path)
    write_meta_sidecar(output_path, metadata)
    logger.info("Saved tokenization cache: {} docs -> {}", len(doc_ids), output_path)


def load_tokenization_cache(path: str | Path) -> tuple[TokenizedLookup, dict[str, Any]]:
    """Load tokenised tensors and reconstruct the doc_id -> tensor mapping."""
    cache_path = Path(path)
    meta_path = cache_path.with_suffix(".meta.json")
    if not cache_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Tokenization cache not found: {cache_path}")

    payload = torch.load(cache_path, weights_only=True)
    metadata = read_meta_sidecar(cache_path)
    input_ids = payload["input_ids"]
    attention_mask = payload["attention_mask"]
    lookup: TokenizedLookup = {
        int(doc_id): {"input_ids": input_ids[i], "attention_mask": attention_mask[i]}
        for i, doc_id in enumerate(payload["doc_ids"].tolist())}
    return lookup, metadata


def tokenize_rows(rows: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase, max_seq_length: int) -> TokenizedLookup:
    """Tokenise a batch of document rows into the per-doc tensor lookup."""
    if not rows:
        return {}
    titles = [str(r.get("title") or "") for r in rows]
    abstracts = [str(r.get("abstract") or "") for r in rows]
    doc_ids = [int(r["doc_id"]) for r in rows]
    encoded = tokenizer(
        titles, abstracts,
        max_length=int(max_seq_length), padding="max_length",
        truncation=True, return_tensors="pt")
    return {
        doc_id: {
            "input_ids": encoded["input_ids"][i].cpu(),
            "attention_mask": encoded["attention_mask"][i].cpu()}
        for i, doc_id in enumerate(doc_ids)}


def build_tokenization_cache(documents: pl.DataFrame, tokenizer: PreTrainedTokenizerBase, max_seq_length: int) -> TokenizedLookup:
    """Tokenise every document once and assemble the doc_id -> tensor lookup."""
    return tokenize_rows(list(documents.iter_rows(named=True)), tokenizer, max_seq_length)


def load_or_build_tokenization_cache(
    documents: pl.DataFrame, tokenizer_name: str, tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int, path: str | Path) -> tuple[TokenizedLookup, dict[str, int]]:
    """Incrementally reuse unchanged token tensors and rebuild new/changed docs."""
    expected = compute_tokenization_metadata(documents, tokenizer_name, max_seq_length)
    current_hashes = coerce_hash_dict(expected)
    rows_by_id = {int(row["doc_id"]): row for row in documents.iter_rows(named=True)}

    old_lookup: TokenizedLookup = {}
    old_hashes: dict[int, str] = {}
    try:
        old_lookup, old_meta = load_tokenization_cache(path)
        if tokenization_is_compatible(old_meta, expected):
            old_hashes = coerce_hash_dict(old_meta)
        else:
            log_cache_miss("tokenization", path, "legacy metadata or tokenizer settings changed")
    except FileNotFoundError:
        log_cache_miss("tokenization", path, "missing cache")

    reused_ids = sorted(
        doc_id for doc_id, h in current_hashes.items()
        if old_hashes.get(doc_id) == h and doc_id in old_lookup)
    rebuild_ids = sorted(set(current_hashes) - set(reused_ids))
    removed_ids = sorted(set(old_lookup) - set(current_hashes))

    lookup: TokenizedLookup = {doc_id: old_lookup[doc_id] for doc_id in reused_ids}
    lookup.update(tokenize_rows([rows_by_id[doc_id] for doc_id in rebuild_ids], tokenizer, max_seq_length))

    # Skip the re-save when nothing changed. Writing 50k+ padded token tensors
    # back to disk on every run is the dominant cost on warm cache hits.
    if rebuild_ids or removed_ids or not old_lookup:
        save_tokenization_cache(lookup, path, expected)
    summary = {"reused": len(reused_ids), "rebuilt": len(rebuild_ids), "removed": len(removed_ids)}
    log_cache_summary("tokenization", path, summary)
    return lookup, summary
