"""
Disk caching for static document embeddings (variant B1).

A frozen pass of the base SciBERT encoder over title + abstract produces a
content-only embedding per document.  These vectors are *not* substituted
into the live `encode_candidates` forward pass — the LoRA adapter trains
through that path and would diverge from any precomputed table — they are
exposed on the run bundle as `doc_embeddings` for downstream consumers
that need a stable, untrained representation (latent graph initialisation,
warm-starting auxiliary heads, retrieval baselines).

The cache is keyed on (model_name, max_seq_length, document fingerprint),
so changing the backbone or sequence length forces a rebuild while reruns
with the same configuration are free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch import Tensor
from transformers import AutoModel, PreTrainedTokenizerBase

from .cache_paths import docs_fingerprint
from .tokenization_cache import TokenizedLookup


def compute_embedding_metadata(
    documents: pl.DataFrame, model_name: str, max_seq_length: int,
) -> dict[str, Any]:
    """Fingerprint the inputs that determine the embedding tensor contents."""
    return {
        "model_name": str(model_name),
        "max_seq_length": int(max_seq_length),
        "num_docs": int(documents.height),
        "docs_fingerprint": docs_fingerprint(documents),
        "pooling": "cls",
    }


def embedding_is_compatible(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Validate that a saved embedding cache matches current settings."""
    return all(metadata.get(k) == expected[k] for k in (
        "model_name", "max_seq_length", "num_docs", "docs_fingerprint", "pooling"
    ))


def save_embedding_cache(
    embeddings: Tensor, doc_ids: list[int],
    path: str | Path, metadata: dict[str, Any],
) -> None:
    """Persist the document embedding matrix and its row-aligned doc ids."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "doc_ids": torch.tensor(doc_ids, dtype=torch.long),
        "embeddings": embeddings.detach().cpu(),
    }, output_path)
    output_path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))


def load_embedding_cache(path: str | Path) -> tuple[Tensor, list[int], dict[str, Any]]:
    """Load the document embedding matrix, the row-aligned ids, and metadata."""
    cache_path = Path(path)
    meta_path = cache_path.with_suffix(".meta.json")
    if not cache_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}")
    payload = torch.load(cache_path, weights_only=True)
    metadata = json.loads(meta_path.read_text())
    doc_ids = [int(v) for v in payload["doc_ids"].tolist()]
    return payload["embeddings"], doc_ids, metadata


def build_embedding_cache(
    documents: pl.DataFrame, model_name: str, max_seq_length: int,
    tokenizer: PreTrainedTokenizerBase,
    tokenized_lookup: TokenizedLookup | None = None,
    batch_size: int = 32, device: str | None = None,
) -> tuple[Tensor, list[int]]:
    """
    Run the frozen base encoder over every document and stack the [CLS] vectors.

    When a `tokenized_lookup` is supplied the tokeniser call is skipped and
    pre-stacked tensors from disk are reused, sharing work with the
    tokenisation cache.  The encoder is loaded fresh, set to eval mode, and
    discarded after the forward pass so it does not occupy GPU memory during
    training.
    """
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = AutoModel.from_pretrained(model_name).to(selected_device).eval()

    doc_ids = [int(d) for d in documents["doc_id"].to_list()]

    if tokenized_lookup is not None and all(d in tokenized_lookup for d in doc_ids):
        input_ids = torch.stack([tokenized_lookup[d]["input_ids"] for d in doc_ids])
        attention_mask = torch.stack([tokenized_lookup[d]["attention_mask"] for d in doc_ids])
    else:
        titles = [str(t) if t is not None else "" for t in documents["title"].to_list()]
        abstracts = [str(a) if a is not None else "" for a in documents["abstract"].to_list()]
        encoded = tokenizer(
            titles, abstracts,
            max_length=int(max_seq_length), padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

    outputs: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, input_ids.size(0), batch_size):
            end = min(start + batch_size, input_ids.size(0))
            batch_ids = input_ids[start:end].to(selected_device)
            batch_mask = attention_mask[start:end].to(selected_device)
            hidden = encoder(input_ids=batch_ids, attention_mask=batch_mask).last_hidden_state
            outputs.append(hidden[:, 0, :].detach().cpu())

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embeddings = torch.cat(outputs, dim=0) if outputs else torch.empty((0, 0))
    return embeddings, doc_ids
