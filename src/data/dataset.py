"""
Dataset and DataLoader helpers for multimodal document + citation-context input.

MultiScaleDocumentDataset converts precomputed citation neighbourhoods into
fixed-shape tensors ready for batching.

The dataset is designed to work with a pre-built neighbor cache so __getitem__
only reads precomputed records and performs tokenisation/tensor assembly, not
graph queries or relevance scoring.

Context neighbours are truncated/padded to max_context_size and tokenised text
is truncated/padded to max_seq_length, so every sample has the same shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .constants import (
    DATALOADER_PREFETCH_FACTOR, REFERENCE_YEAR, UNKNOWN_TOKEN, YEAR,
    YEAR_SCALE, EdgeType, NeighborCache)


def create_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """
    Create the HF tokenizer shared by center-document and context text paths.

    A single tokenizer instance keeps vocabulary and special-token settings
    identical for all center and neighbour inputs.
    """
    return AutoTokenizer.from_pretrained(model_name)


def as_flat_string_list(values: Any) -> list[str]:
    """Normalise nested Polars/list author values to a flat list of strings."""
    if hasattr(values, "to_list"):
        values = values.to_list()
    if not isinstance(values, list):
        return []

    flat: list[str] = []
    for value in values:
        if hasattr(value, "to_list"):
            flat.extend(str(item) for item in value.to_list() if item is not None)
        elif isinstance(value, list):
            flat.extend(str(item) for item in value if item is not None)
        elif value is not None:
            flat.append(str(value))

    return flat


def pad_list(values: list[Any], size: int, pad_value: Any) -> list[Any]:
    """Return values truncated/padded to an exact fixed length."""
    return values[:size] + [pad_value] * max(0, size - len(values))


class MultiScaleDocumentDataset(Dataset):
    """
    Expose document samples with metadata and fixed-size context feature blocks.

    Each sample corresponds to one center document and its pre-ranked citation
    neighbours from the context cache.

    Returned sample fields:
    - center text: input_ids, attention_mask
    - metadata: venue_ids, publisher_ids, author_ids, years
    - context text: context_input_ids, context_attention_mask
    - context structure: edge type, year delta, score, venue/publisher ids, years
    - optional context_hop_profiles and context_spectral
    - optional labels for supervised rows

    Expensive graph work is expected to have already happened during cache
    construction. __getitem__ only performs tokenisation and tensor assembly.
    """

    # Year normalisation constants are shared with the caching layer.
    REFERENCE_YEAR: float = REFERENCE_YEAR
    YEAR_SCALE: float = YEAR_SCALE

    def __init__(
        self, documents: pl.DataFrame, tokenizer: PreTrainedTokenizerBase,
        venue_encoder: dict[str, int], publisher_encoder: dict[str, int],
        author_encoder: dict[str, int], max_seq_length: int,
        max_context_size: int, max_authors: int,
        context_documents: pl.DataFrame | None = None,
        context_cache: NeighborCache | None = None,
        cache_text: bool = True, pretokenize_context: bool = False,
        hop_profile_dim: int = 2, spectral_dim: int = 0,
        pretokenized: dict[int, dict[str, Tensor]] | None = None) -> None:
        """
        Initialise lookup structures and optional in-memory token caches.

        Frames are cloned so external mutations do not silently change dataset
        state. Doc-id-keyed dicts are built up front because repeated row access
        by integer index is much slower inside __getitem__.
        """
        self.documents = documents.clone()
        self.context_documents = context_documents.clone() if context_documents is not None else documents.clone()
        self.tokenizer = tokenizer
        self.venue_encoder = venue_encoder
        self.publisher_encoder = publisher_encoder
        self.author_encoder = author_encoder
        self.max_seq_length = int(max_seq_length)
        self.max_context_size = int(max_context_size)
        self.max_authors = int(max_authors)
        self.hop_profile_dim = int(hop_profile_dim)
        self.spectral_dim = int(spectral_dim)
        self.cache_text = bool(cache_text)
        self.context_cache = context_cache or {}

        # Flat dict lookup for neighbour document rows; faster than row-by-index.
        self.doc_lookup: dict[int, dict[str, Any]] = {int(row["doc_id"]): row for row in self.context_documents.iter_rows(named=True)}

        # Seed text cache from disk-backed tokenisation when available. The dataset
        # still owns this dict so newly encountered docs can be cached locally.
        self.text_cache: dict[int, dict[str, Tensor]] = dict(pretokenized) if pretokenized else {}

        # Reused for padding slots to avoid repeated zero tensor allocations.
        self.empty_text = self.build_empty_neighbor_text()

        if self.cache_text and pretokenize_context:
            self.pretokenize_context_documents()

    def __len__(self) -> int:
        """Return the number of center documents in this dataset split."""
        return self.documents.height

    def pretokenize_context_documents(self) -> None:
        """
        Warm the in-memory text cache for every available context document.

        This removes tokeniser overhead during training at the cost of RAM
        proportional to context document count times max_seq_length.
        """
        for doc_id, row in self.doc_lookup.items():
            self.text_cache[doc_id] = self.tokenize_text(str(row.get("title", "")), str(row.get("abstract", "")))

    def tokenize_text(self, title: str, abstract: str) -> dict[str, Tensor]:
        """
        Tokenise a title/abstract pair into the canonical input_ids layout.

        The tokenizer concatenates title and abstract with its standard separator,
        then truncates/pads so all tensors in a batch have the same shape.
        """
        encoded = self.tokenizer(
            title, abstract, max_length=self.max_seq_length, padding="max_length",
            truncation=True, return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)}

    def tokenized_text(self, doc_id: int, title: str, abstract: str) -> dict[str, Tensor]:
        """
        Return tokenised text, reading from the in-memory cache when available.

        When cache_text is disabled, every call re-tokenises, which can be useful
        when memory is constrained and the tokenizer is fast enough.
        """
        if self.cache_text and doc_id in self.text_cache:
            return self.text_cache[doc_id]

        tokenized = self.tokenize_text(title, abstract)
        if self.cache_text:
            self.text_cache[doc_id] = tokenized
        return tokenized

    def build_empty_neighbor_text(self) -> dict[str, Tensor]:
        """
        Create the reusable zero-tensor pair used for padded neighbour slots.

        The object is read-only in practice, so stacking many references to it is
        cheaper than allocating new zero tensors per sample.
        """
        return {
            "input_ids": torch.zeros(self.max_seq_length, dtype=torch.long),
            "attention_mask": torch.zeros(self.max_seq_length, dtype=torch.long)}

    def parse_year(self, value: Any, default: float = float(YEAR)) -> float:
        """
        Convert a year-like value to float with a stable fallback.

        Invalid forms collapse to default so NaN never propagates into model features.
        """
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def normalize_year(self, year: float) -> float:
        """Map a raw publication year into a small centred feature range."""
        return (year - self.REFERENCE_YEAR) / self.YEAR_SCALE

    def metadata_struct(self, row: dict[str, Any]) -> dict[str, Tensor]:
        """
        Build fixed-shape metadata tensors for a single document.

        Authors are truncated to max_authors and padded with zero. Venue and
        publisher fall back to UNKNOWN_TOKEN so missing values use a consistent
        category rather than raising KeyError.
        """
        authors = as_flat_string_list(row["authors"])[:self.max_authors]
        author_ids = [self.author_encoder.get(author, 0) for author in authors]
        author_ids = pad_list(author_ids, self.max_authors, 0)

        return {
            "author_ids": torch.tensor(author_ids, dtype=torch.long),
            "venue_ids": torch.tensor(
                self.venue_encoder.get(str(row.get("venue", UNKNOWN_TOKEN)), 0), dtype=torch.long),
            "publisher_ids": torch.tensor(
                self.publisher_encoder.get(str(row.get("publisher", UNKNOWN_TOKEN)), 0), dtype=torch.long),
            "years": torch.tensor([
                self.normalize_year(self.parse_year(row.get("year")))], dtype=torch.float32)}

    def context_entry_tensors(self, center_year: float, entry: dict[str, Any]) -> dict[str, Any] | None:
        """
        Convert one cache entry into normalised, tensor-ready pieces.

        Returns None when the neighbour row cannot be found, e.g. because it was
        pruned from this split. Year delta is recomputed from raw years so the
        dataset remains aligned with its current normalisation constants.
        """
        neighbor_id = int(entry["doc_id"])
        neighbor_row = self.doc_lookup.get(neighbor_id)
        if neighbor_row is None:
            return None

        neighbor_year = self.parse_year(neighbor_row["year"], default=center_year)
        result: dict[str, Any] = {
            "text": self.tokenized_text(
                neighbor_id, str(neighbor_row["title"]), str(neighbor_row["abstract"])),
            "edge_type": int(entry.get("edge_type", int(EdgeType.NONE))),
            "venue_id": self.venue_encoder.get(str(neighbor_row.get("venue", UNKNOWN_TOKEN)), 0),
            "publisher_id": self.publisher_encoder.get(str(neighbor_row.get("publisher", UNKNOWN_TOKEN)), 0),
            "year": self.normalize_year(neighbor_year),
            "year_delta": (neighbor_year - center_year) / YEAR_SCALE,
            "score": float(entry["score"])}

        # Optional structural features are truncated/padded to their configured dimensions.
        if self.hop_profile_dim > 0 and (hop_profile := entry.get("hop_profile")):
            result["hop_profile"] = pad_list(list(hop_profile), self.hop_profile_dim, 0.0)
        if self.spectral_dim > 0 and (spectral := entry.get("spectral")):
            result["spectral"] = pad_list(list(spectral), self.spectral_dim, 0.0)

        return result

    def empty_context_lists(self) -> dict[str, list[Any]]:
        """Create parallel lists used to accumulate neighbour tensors before stacking."""
        return {
            "neighbors": [], "edge_types": [], "year_deltas": [], "scores": [],
            "hop_profiles": [], "spectral": [], "venue_ids": [], "publisher_ids": [], "years": []}

    def pad_context_lists(self, values: dict[str, list[Any]], pad_count: int) -> None:
        """Pad accumulated context lists in-place to max_context_size."""
        values["neighbors"] += [self.empty_text] * pad_count
        values["edge_types"] += [int(EdgeType.NONE)] * pad_count
        values["year_deltas"] += [0.0] * pad_count
        values["scores"] += [0.0] * pad_count
        values["venue_ids"] += [0] * pad_count
        values["publisher_ids"] += [0] * pad_count
        values["years"] += [0.0] * pad_count

        if self.hop_profile_dim > 0:
            values["hop_profiles"] += [[0.0] * self.hop_profile_dim] * pad_count
        if self.spectral_dim > 0:
            values["spectral"] += [[0.0] * self.spectral_dim] * pad_count

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """
        Build one multimodal sample with center-document and context features.

        Main shapes:
        - input_ids: (max_seq_length,)
        - author_ids: (max_authors,)
        - context_input_ids: (max_context_size, max_seq_length)
        - context_mask: (max_context_size,)
        - optional context_hop_profiles: (max_context_size, hop_profile_dim)
        - optional context_spectral: (max_context_size, spectral_dim)
        """
        row = self.documents.row(idx, named=True)
        doc_id = int(row["doc_id"])
        center_year = self.parse_year(row["year"])

        item: dict[str, Tensor] = {
            "doc_id": torch.tensor(doc_id, dtype=torch.long),
            **self.tokenized_text(doc_id, str(row["title"]), str(row["abstract"])),
            **self.metadata_struct(row)}

        values = self.empty_context_lists()
        for entry in self.context_cache.get(doc_id, [])[:self.max_context_size]:
            tensors = self.context_entry_tensors(center_year, entry)
            if tensors is None:
                continue

            # Neighbour text tensors are the heaviest part, so unresolved rows are skipped early.
            values["neighbors"].append(tensors["text"])
            values["edge_types"].append(tensors["edge_type"])
            values["year_deltas"].append(tensors["year_delta"])
            values["scores"].append(tensors["score"])
            values["venue_ids"].append(tensors["venue_id"])
            values["publisher_ids"].append(tensors["publisher_id"])
            values["years"].append(tensors["year"])

            if self.hop_profile_dim > 0:
                values["hop_profiles"].append(tensors.get("hop_profile", [0.0] * self.hop_profile_dim))
            if self.spectral_dim > 0:
                values["spectral"].append(tensors.get("spectral", [0.0] * self.spectral_dim))

        valid_count = len(values["neighbors"])
        pad_count = max(0, self.max_context_size - valid_count)
        self.pad_context_lists(values, pad_count)

        item["context_input_ids"] = torch.stack([text["input_ids"] for text in values["neighbors"]])
        item["context_attention_mask"] = torch.stack([text["attention_mask"] for text in values["neighbors"]])
        item["context_mask"] = torch.tensor([1] * valid_count + [0] * pad_count, dtype=torch.long)
        item["context_edge_types"] = torch.tensor(values["edge_types"], dtype=torch.long)
        item["context_year_deltas"] = torch.tensor(values["year_deltas"], dtype=torch.float32)
        item["context_scores"] = torch.tensor(values["scores"], dtype=torch.float32)
        item["context_venue_ids"] = torch.tensor(values["venue_ids"], dtype=torch.long)
        item["context_publisher_ids"] = torch.tensor(values["publisher_ids"], dtype=torch.long)
        item["context_years"] = torch.tensor(values["years"], dtype=torch.float32)

        if self.hop_profile_dim > 0:
            item["context_hop_profiles"] = torch.tensor(values["hop_profiles"], dtype=torch.float32)
        if self.spectral_dim > 0:
            item["context_spectral"] = torch.tensor(values["spectral"], dtype=torch.float32)

        # Unlabeled rows omit labels so the model/training loop can skip loss computation.
        label = row["label"]
        if label is not None and not (isinstance(label, float) and np.isnan(label)):
            item["labels"] = torch.tensor(int(label), dtype=torch.long)

        return item


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 1) -> DataLoader:
    """
    Build a DataLoader with CUDA-aware pinned memory and worker-only prefetching.

    Worker persistence and prefetching are enabled only when worker processes
    exist, because those settings are meaningless in single-process loading.
    """
    kwargs: dict[str, Any] = {
        "dataset": dataset, "batch_size": batch_size, "shuffle": shuffle,
        "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}

    if num_workers > 0:
        kwargs |= {"persistent_workers": True, "prefetch_factor": DATALOADER_PREFETCH_FACTOR}

    return DataLoader(**kwargs)
