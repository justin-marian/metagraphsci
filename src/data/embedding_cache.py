"""Incremental disk caching for static document embeddings (variant B1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import torch
from loguru import logger
from torch import Tensor
from transformers import AutoModel, PreTrainedTokenizerBase

from .cache_utils import (
    coerce_hash_dict, docs_fingerprint, log_cache_miss, log_cache_summary,
    metadata_matches, per_doc_hashes, read_meta_sidecar, write_meta_sidecar)
from .tokenization_cache import TokenizedLookup

POOLING_CLS = "cls"

COMPATIBILITY_KEYS = ("model_name", "max_seq_length", "pooling")


def compute_embedding_metadata(documents: pl.DataFrame, model_name: str, max_seq_length: int, pooling: str = POOLING_CLS) -> dict[str, Any]:
    """Fingerprint the inputs that determine the embedding tensor contents."""
    return {
        "model_name": str(model_name),
        "max_seq_length": int(max_seq_length),
        "num_docs": int(documents.height),
        "docs_fingerprint": docs_fingerprint(documents),
        "per_doc_hashes": per_doc_hashes(documents),
        "pooling": str(pooling)}


def embedding_is_compatible(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return True when model settings match and per-document hashes exist."""
    return (
        metadata_matches("embedding", metadata, expected, COMPATIBILITY_KEYS)
        and isinstance(metadata.get("per_doc_hashes"), dict))


def save_embedding_cache(embeddings: Tensor, doc_ids: list[int], path: str | Path, metadata: dict[str, Any]) -> None:
    """Persist the document embedding matrix and its row-aligned doc ids."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "doc_ids": torch.tensor(doc_ids, dtype=torch.long),
        "embeddings": embeddings.detach().cpu()}, output_path)
    write_meta_sidecar(output_path, metadata)
    logger.info("Saved embedding cache: {} docs -> {}", len(doc_ids), output_path)


def load_embedding_cache(path: str | Path) -> tuple[Tensor, list[int], dict[str, Any]]:
    """Load the document embedding matrix, the row-aligned ids, and metadata."""
    cache_path = Path(path)
    meta_path = cache_path.with_suffix(".meta.json")
    if not cache_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}")
    payload = torch.load(cache_path, weights_only=True)
    metadata = read_meta_sidecar(cache_path)
    doc_ids = [int(v) for v in payload["doc_ids"].tolist()]
    return payload["embeddings"], doc_ids, metadata


def pool_hidden(hidden: Tensor, attention_mask: Tensor, pooling: str) -> Tensor:
    """Pool last hidden states according to the stored cache strategy."""
    if pooling == POOLING_CLS:
        return hidden[:, 0, :]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    raise ValueError(f"Unsupported pooling strategy: {pooling!r}")


def token_tensors_for_docs(
    documents: pl.DataFrame, doc_ids: list[int],
    tokenizer: PreTrainedTokenizerBase, tokenized_lookup: TokenizedLookup | None,
    max_seq_length: int) -> tuple[Tensor, Tensor]:
    """Return stacked input ids and masks for a doc-id ordered subset."""
    if not doc_ids:
        return torch.empty((0, max_seq_length), dtype=torch.long), torch.empty((0, max_seq_length), dtype=torch.long)

    if tokenized_lookup is not None and all(doc_id in tokenized_lookup for doc_id in doc_ids):
        return (
            torch.stack([tokenized_lookup[d]["input_ids"] for d in doc_ids]),
            torch.stack([tokenized_lookup[d]["attention_mask"] for d in doc_ids]))

    rows = {int(r["doc_id"]): r for r in documents.iter_rows(named=True)}
    titles = [str(rows[d].get("title") or "") for d in doc_ids]
    abstracts = [str(rows[d].get("abstract") or "") for d in doc_ids]
    encoded = tokenizer(
        titles, abstracts,
        max_length=int(max_seq_length), padding="max_length",
        truncation=True, return_tensors="pt")
    return encoded["input_ids"], encoded["attention_mask"]


def encode_doc_ids(
    documents: pl.DataFrame, doc_ids: list[int],
    model_name: str, max_seq_length: int, tokenizer: PreTrainedTokenizerBase,
    tokenized_lookup: TokenizedLookup | None,
    batch_size: int, device: str | None, pooling: str) -> Tensor:
    """Run the frozen base encoder for a doc-id ordered subset."""
    if not doc_ids:
        return torch.empty((0, 0))
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = AutoModel.from_pretrained(model_name).to(selected_device).eval()
    input_ids, attention_mask = token_tensors_for_docs(documents, doc_ids, tokenizer, tokenized_lookup, max_seq_length)

    outputs: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, input_ids.size(0), batch_size):
            end = min(start + batch_size, input_ids.size(0))
            batch_ids = input_ids[start:end].to(selected_device)
            batch_mask = attention_mask[start:end].to(selected_device)
            hidden = encoder(input_ids=batch_ids, attention_mask=batch_mask).last_hidden_state
            outputs.append(pool_hidden(hidden, batch_mask, pooling).detach().cpu())

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 0))


def build_embedding_cache(
    documents: pl.DataFrame, model_name: str, max_seq_length: int,
    tokenizer: PreTrainedTokenizerBase,
    tokenized_lookup: TokenizedLookup | None = None,
    batch_size: int = 32, device: str | None = None,
    pooling: str = POOLING_CLS) -> tuple[Tensor, list[int]]:
    """Run the frozen base encoder over every document and preserve doc alignment."""
    doc_ids = [int(d) for d in documents["doc_id"].to_list()]
    return (
        encode_doc_ids(documents, doc_ids, model_name, max_seq_length, tokenizer, tokenized_lookup, batch_size, device, pooling),
        doc_ids)


def load_or_build_embedding_cache(
    documents: pl.DataFrame, model_name: str,
    max_seq_length: int, tokenizer: PreTrainedTokenizerBase,
    path: str | Path, tokenized_lookup: TokenizedLookup | None = None,
    batch_size: int = 32, device: str | None = None, pooling: str = POOLING_CLS
) -> tuple[Tensor, list[int], dict[str, int]]:
    """Incrementally reuse unchanged embeddings and recompute new/changed docs."""
    expected = compute_embedding_metadata(documents, model_name, max_seq_length, pooling=pooling)
    current_hashes = coerce_hash_dict(expected)
    current_doc_ids = [int(d) for d in documents["doc_id"].to_list()]

    old_by_id: dict[int, Tensor] = {}
    old_hashes: dict[int, str] = {}
    try:
        old_embeddings, old_doc_ids, old_meta = load_embedding_cache(path)
        if embedding_is_compatible(old_meta, expected):
            old_hashes = coerce_hash_dict(old_meta)
            old_by_id = {doc_id: old_embeddings[i] for i, doc_id in enumerate(old_doc_ids)}
        else:
            log_cache_miss("embedding", path, "legacy metadata or model settings changed")
    except FileNotFoundError:
        log_cache_miss("embedding", path, "missing cache")

    # Use a set for the membership check, otherwise the per-row ``in reused_ids``
    # turns the assembly loop into O(N^2) on warm cache hits.
    reused_set = {d for d in current_doc_ids if old_hashes.get(d) == current_hashes[d] and d in old_by_id}
    reused_ids = [d for d in current_doc_ids if d in reused_set]
    rebuild_ids = [d for d in current_doc_ids if d not in reused_set]
    removed_ids = sorted(set(old_by_id) - set(current_doc_ids))

    rebuilt = encode_doc_ids(
        documents, rebuild_ids, model_name, max_seq_length, tokenizer,
        tokenized_lookup, batch_size, device, pooling)
    rebuilt_by_id = {doc_id: rebuilt[i] for i, doc_id in enumerate(rebuild_ids)}

    rows = [old_by_id[doc_id] if doc_id in reused_set else rebuilt_by_id[doc_id] for doc_id in current_doc_ids]
    embeddings = torch.stack(rows) if rows else torch.empty((0, 0))

    # Skip the re-save when nothing changed. Writing the full embedding matrix
    # back to disk on every warm run is the dominant cost on cache hits.
    if rebuild_ids or removed_ids or not old_by_id:
        save_embedding_cache(embeddings, current_doc_ids, path, expected)
    summary = {"reused": len(reused_ids), "rebuilt": len(rebuild_ids), "removed": len(removed_ids)}
    log_cache_summary("embedding", path, summary)
    return embeddings, current_doc_ids, summary
