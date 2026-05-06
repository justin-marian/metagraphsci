"""
Disk caching for pre-tokenised document text (title + abstract).

Tokenisation results depend only on (tokenizer, max_seq_length, document
text), so they are seed-independent and live under the `global` cache scope.
The cache stores stacked `input_ids` / `attention_mask` tensors plus the
matching `doc_ids` so the dataset layer can build a doc_id -> tensor lookup
without any tokeniser calls during training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .cache_paths import docs_fingerprint


TokenizedLookup = dict[int, dict[str, Tensor]]


def compute_tokenization_metadata(
    documents: pl.DataFrame, tokenizer_name: str, max_seq_length: int,
) -> dict[str, Any]:
    """Fingerprint inputs that determine the tokenised tensor contents."""
    return {
        "tokenizer_name": str(tokenizer_name),
        "max_seq_length": int(max_seq_length),
        "num_docs": int(documents.height),
        "docs_fingerprint": docs_fingerprint(documents),
    }


def tokenization_is_compatible(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Validate that a saved tokenisation cache matches current settings."""
    return all(metadata.get(k) == expected[k] for k in (
        "tokenizer_name", "max_seq_length", "num_docs", "docs_fingerprint"
    ))


def save_tokenization_cache(
    lookup: TokenizedLookup, path: str | Path, metadata: dict[str, Any],
) -> None:
    """Persist tokenised tensors as a single torch file plus a JSON sidecar."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc_ids = sorted(lookup.keys())
    if doc_ids:
        input_ids = torch.stack([lookup[d]["input_ids"] for d in doc_ids])
        attention_mask = torch.stack([lookup[d]["attention_mask"] for d in doc_ids])
    else:
        input_ids = torch.empty((0, int(metadata["max_seq_length"])), dtype=torch.long)
        attention_mask = torch.empty((0, int(metadata["max_seq_length"])), dtype=torch.long)

    torch.save({
        "doc_ids": torch.tensor(doc_ids, dtype=torch.long),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, output_path)
    output_path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))


def load_tokenization_cache(path: str | Path) -> tuple[TokenizedLookup, dict[str, Any]]:
    """Load tokenised tensors and reconstruct the doc_id -> tensor mapping."""
    cache_path = Path(path)
    meta_path = cache_path.with_suffix(".meta.json")
    if not cache_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Tokenization cache not found: {cache_path}")

    payload = torch.load(cache_path, weights_only=True)
    metadata = json.loads(meta_path.read_text())

    doc_ids = payload["doc_ids"].tolist()
    input_ids = payload["input_ids"]
    attention_mask = payload["attention_mask"]
    lookup: TokenizedLookup = {
        int(doc_id): {
            "input_ids": input_ids[i],
            "attention_mask": attention_mask[i],
        }
        for i, doc_id in enumerate(doc_ids)
    }
    return lookup, metadata


def build_tokenization_cache(
    documents: pl.DataFrame, tokenizer: PreTrainedTokenizerBase, max_seq_length: int,
) -> TokenizedLookup:
    """Tokenise every document once and assemble the doc_id -> tensor lookup."""
    titles = [str(t) if t is not None else "" for t in documents["title"].to_list()]
    abstracts = [str(a) if a is not None else "" for a in documents["abstract"].to_list()]
    doc_ids = [int(d) for d in documents["doc_id"].to_list()]

    encoded = tokenizer(
        titles, abstracts,
        max_length=int(max_seq_length), padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    return {
        doc_id: {
            "input_ids": input_ids[i],
            "attention_mask": attention_mask[i],
        }
        for i, doc_id in enumerate(doc_ids)
    }
