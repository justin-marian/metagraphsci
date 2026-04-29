"""
Dataset and DataLoader helpers for multimodal document + citation-context input.

MultiScaleDocumentDataset converts precomputed citation neighbourhoods 
(produced by context cache construction) into fixed-shape tensors ready for batching.

The dataset is designed to work with a pre-built neighbor cache so __getitem__ only 
needs to perform tokenisation and tensor assembly, not expensive graph queries or relevance scoring.

Fixed-size tensor blocks so a standard PyTorch DataLoader can feed the model
without any online graph sampling. Context neighbours are truncated/padded to 
max_context_size and tokenised text is truncated/padded to max_seq_length,
so every sample has the same shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from constants import (
    DATALOADER_PREFETCH_FACTOR,
    YEAR, REFERENCE_YEAR,
    UNKNOWN_TOKEN, YEAR_SCALE,
    EdgeType, NeighborCache)


def create_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """
    Create the HF tokenizer shared by center-document and context text paths.

    A single tokenizer instance is shared across the dataset so the vocabulary
    and special-token settings are guaranteed to be identical for all inputs.
    """
    return AutoTokenizer.from_pretrained(model_name)


class MultiScaleDocumentDataset(Dataset):
    """
    Expose document samples including metadata and fixed-size blocks of context features.
    Each sample corresponds to one center document and its pre-ranked list of citation
    neighbours from the context cache.  

    The dataset is designed to work with a pre-built cache so __getitem__ 
    only needs to perform tokenisation and tensor assembly, not expensive graph queries or relevance scoring.

    Each sample returned contains:
    ------------------------------

    - Tokenised title + abstract of the center document.
    - Metadata tensors: venue_id, publisher_id, author_ids, year.

    - A fixed-length block of context_input_ids / context_attention_mask for
      up to max_context_size neighbours (padded with zeros when fewer are available).

    - Structural context features: edge type, year delta, score, venue/publisher
      ids, hop profile, and optional spectral coordinates for each neighbour.

    - A context_mask indicating which neighbour slots contain real data.
    - An optional 'labels' tensor for supervised training.

    Expensive graph work is expected to have already happened during cache construction.  
    __getitem__ only performs tokenisation (optionally cached) and tensor assembly.
    """

    # Year normalisation constants: caching layer are guaranteed to use identical scaling.
    REFERENCE_YEAR: float = REFERENCE_YEAR   # Center point of the year feature range.
    YEAR_SCALE:     float = YEAR_SCALE       # Maps ±26 yr window => roughly [−1, 1]

    def __init__(
        self,
        documents:          pl.DataFrame,
        tokenizer:          PreTrainedTokenizerBase,
        venue_encoder:      dict[str, int],
        publisher_encoder:  dict[str, int],
        author_encoder:     dict[str, int],
        max_seq_length:     int,
        max_context_size:   int,
        max_authors:        int,
        context_documents:  pl.DataFrame | None = None,
        context_cache:      NeighborCache | None = None,
        cache_text:         bool = True,
        pretokenize_context: bool = False,
        hop_profile_dim:    int = 2,
        spectral_dim:       int = 0) -> None:
        """Initialise lookup structures and optional in-memory token caches.

        Parameters
        ----------
        documents           : Center-document rows used for __getitem__ indexing.
        tokenizer           : HuggingFace tokenizer shared across all text paths.

        venue_encoder       : {venue_name => int id} mapping.
        publisher_encoder   : {publisher_name => int id} mapping.
        author_encoder      : {author_name => int id} mapping

        Maximum sizes and truncation settings
        --------------------------------------
        max_seq_length      : Truncation length for title+abstract tokenisation.
        max_context_size    : Number of neighbour slots per sample (fixed).
        max_authors         : Maximum number of author ids per document.

        Context parameters
        --------------------
        context_documents   : Document table used for neighbour text lookups.
                              Defaults to documents when None (same split).
        context_cache       : Pre-built neighbour cache from build_neighbor_cache.
                              Empty dict means every sample has zero neighbours.
        cache_text          : If True, cache tokenised tensors in RAM on first use.
        pretokenize_context : If True and cache_text is True, warm the text cache
                              for all context documents at init time.  Trades RAM
                              for lower CPU pressure in the training loop.

        Hop profile and spectral features
        ----------------------------------
        hop_profile_dim     : Number of BFS-hop counts stored per neighbour (0 = off).
        spectral_dim        : Laplacian eigenvector dimension per neighbour (0 = off).

        Implementation notes
        --------------------
        Frames are cloned so later external mutations do not silently change the
        dataset state.  Doc-id-keyed dicts are built up front because repeated
        row access by integer index is much slower than direct dictionary lookup
        inside __getitem__.
        """
        self.documents          = documents.clone()
        self.context_documents  = context_documents.clone() if context_documents is not None else documents.clone()
        self.tokenizer          = tokenizer
        self.venue_encoder      = venue_encoder
        self.publisher_encoder  = publisher_encoder
        self.author_encoder     = author_encoder
        self.max_seq_length     = int(max_seq_length)
        self.max_context_size   = int(max_context_size)
        self.max_authors        = int(max_authors)
        self.hop_profile_dim    = int(hop_profile_dim)
        self.spectral_dim       = int(spectral_dim)
        self.cache_text         = bool(cache_text)
        self.context_cache      = context_cache or {}

        # Flat dict lookup for neighbour document rows; faster than row-by-index.
        self.doc_lookup: dict[int, dict[str, Any]] = {
            int(row["doc_id"]): row
            for row in self.context_documents.iter_rows(named=True)
        }
        self.text_cache: dict[int, dict[str, Tensor]] = {}
        # A pre-built zero-tensor pair reused for all padding slots, avoiding
        # repeated zero tensor allocations inside the hot __getitem__ path.
        self.empty_text = self.build_empty_neighbor_text()

        if self.cache_text and pretokenize_context:
            self.pretokenize_context_documents()

    def __len__(self) -> int:
        """Return the number of center documents in this dataset split."""
        return self.documents.height

    def pretokenize_context_documents(self) -> None:
        """
        Warm the in-memory text cache for every available context document.

        Calling this at init time converts all title+abstract pairs to tensors
        once, eliminating tokeniser overhead during training. The trade-off is
        higher memory usage proportional to the number of context documents times max_seq_length.
        """
        for doc_id, row in self.doc_lookup.items():
            self.text_cache[doc_id] = self.tokenize_text(str(row.get("title", "")), str(row.get("abstract", "")))

    def tokenize_text(self, title: str, abstract: str) -> dict[str, Tensor]:
        """
        Tokenise a title/abstract pair into the canonical input_ids layout.

        The tokeniser concatenates title and abstract with its standard
        separator, truncates to max_seq_length, and pads shorter texts so all
        tensors in a batch have the same shape.
        """
        encoded = self.tokenizer(
            title, abstract,
            max_length=self.max_seq_length, padding="max_length",
            truncation=True, return_tensors="pt")
        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }

    def tokenized_text(self, doc_id: int, title: str, abstract: str) -> dict[str, Tensor]:
        """
        Return tokenised text, reading from the in-memory cache when available.

        Cache behaviour is controlled by cached text. When disabled, every call re-tokenises 
        (useful when memory is constrained and the tokeniser is fast).
        """
        if self.cache_text and doc_id in self.text_cache:
            return self.text_cache[doc_id]
        tokenized = self.tokenize_text(title, abstract)
        if self.cache_text:
            self.text_cache[doc_id] = tokenized
        return tokenized

    def build_empty_neighbor_text(self) -> dict[str, Tensor]:
        """
        Create the reusable zero-tensor pair used for all padded neighbour slots.

        A shared empty-text object is preferred over allocating a new pair for
        every padding slot because the object is read-only and stacking many
        references to it is cheap.
        """
        return {
            "input_ids":      torch.zeros(self.max_seq_length, dtype=torch.long),
            "attention_mask": torch.zeros(self.max_seq_length, dtype=torch.long)
        }

    def parse_year(self, value: Any, default: float = float(YEAR)) -> float:
        """
        Convert a year-like value to float with a stable fallback.

        Metadata sources are heterogeneous: values may be integers, float NaN,
        None, or unparseable strings. All invalid forms collapse to the
        provided default so NaN never propagates into model features.
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

        Authors are truncated to max_authors and padded with zeros so every
        sample in a batch has an author_ids tensor of identical length.

        Venue and publisher fall back to the UNKNOWN_TOKEN when the string is absent 
        from the row so missing values are treated as a consistent category rather than crashing with a KeyError.
        """
        authors = row["authors"]
        # Guard against non-list encodings that may arrive from older CSV rows.
        authors = authors[:self.max_authors] if isinstance(authors, list) else []
        author_ids = [self.author_encoder[author] for author in authors]
        # Pad the author list to a fixed length with 0 (the unknown/padding index).
        author_ids.extend([0] * max(0, self.max_authors - len(author_ids)))

        return {
            "venue_ids":     torch.tensor(self.venue_encoder[str(row.get("venue", UNKNOWN_TOKEN))], dtype=torch.long),
            "publisher_ids": torch.tensor(self.publisher_encoder[(str(row.get("publisher", UNKNOWN_TOKEN)))], dtype=torch.long),
            "author_ids":    torch.tensor(author_ids[:self.max_authors], dtype=torch.long),
            "years":         torch.tensor([self.normalize_year(self.parse_year(row.get("year")))], dtype=torch.float32)
        }

    def context_entry_tensors(self, center_year: float, entry: dict[str, Any]) -> dict[str, Any] | None:
        """
        Convert one cache entry into normalised, tensor-ready pieces.

        Returns None when the neighbour's document row cannot be found in documents lookup
        (e.g. because it was pruned from this split or was never downloaded), 
        so the caller can skip this entry without crashing.

        Year delta is re-computed from raw years rather than trusting the cached year_delta field. 
        This ensures the dataset stays correct even if the normalisation change due to a different 
        REFERENCE_YEAR or YEAR_SCALE, as long as the cache was built with the same normalisation constants.
        """
        neighbor_id  = int(entry["doc_id"])
        neighbor_row = self.doc_lookup[neighbor_id]
        # The neighbour was pruned from this split or was never downloaded.
        if neighbor_row is None:
            return None

        neighbor_year = self.parse_year(neighbor_row["year"], default=center_year)

        result: dict[str, Any] = {
            "text":         self.tokenized_text(neighbor_id, str(neighbor_row["title"]), str(neighbor_row["abstract"])),
            "edge_type":    int(entry.get("edge_type", int(EdgeType.NONE))),
            "venue_id":     self.venue_encoder[str(neighbor_row.get("venue", UNKNOWN_TOKEN))],
            "publisher_id": self.publisher_encoder[str(neighbor_row.get("publisher", UNKNOWN_TOKEN))],
            "year":         self.normalize_year(neighbor_year),
            "year_delta":   (neighbor_year - center_year) / YEAR_SCALE,
            "score":        float(entry["score"])
        }

        # Hop profile: truncate/pad to exactly hop_profile_dim floats.
        if self.hop_profile_dim > 0 and (hp := entry.get("hop_profile")):
            result["hop_profile"] = hp[:self.hop_profile_dim] + [0.0] * max(0, self.hop_profile_dim - len(hp))

        # Spectral coordinates: same truncate/pad logic.
        if self.spectral_dim > 0 and (sp := entry.get("spectral")):
            result["spectral"] = sp[:self.spectral_dim] + [0.0] * max(0, self.spectral_dim - len(sp))

        return result

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Build one multimodal sample with centre-document and context features.

        Structure of the returned dict
        --------------------------------
        doc_id                 – scalar int64
        input_ids              – (max_seq_length,) int64 center text tokens
        attention_mask         – (max_seq_length,) int64
        venue_ids              – scalar int64
        publisher_ids          – scalar int64
        author_ids             – (max_authors,) int64
        years                  – (1,) float32
        context_input_ids      – (max_context_size, max_seq_length) int64
        context_attention_mask – (max_context_size, max_seq_length) int64
        context_mask           – (max_context_size,) int64  1=real 0=pad
        context_edge_types     – (max_context_size,) int64
        context_year_deltas    – (max_context_size,) float32
        context_scores         – (max_context_size,) float32
        context_venue_ids      – (max_context_size,) int64
        context_publisher_ids  – (max_context_size,) int64
        context_years          – (max_context_size,) float32

        Options for additional context features:
        -----------------------------------------
        context_hop_profiles   – (max_context_size, hop_profile_dim) float32  [optional]
        context_spectral       – (max_context_size, spectral_dim) float32     [optional]
        labels                 – scalar int64                                 [optional]
        """
        row = self.documents.row(idx, named=True)
        doc_id = int(row["doc_id"])
        center_year = self.parse_year(row["year"])

        # Start the item dict with center-document tensors.
        item: dict[str, Tensor] = {
            "doc_id": torch.tensor(doc_id, dtype=torch.long),
            **self.tokenized_text(doc_id, str(row["title"]), str(row["abstract"])),
            **self.metadata_struct(row)
        }

        # Accumulate neighbour tensors in parallel lists before stacking.
        neighbors:      list[dict[str, Tensor]] = []
        edge_types:     list[int]   = []
        year_deltas:    list[float] = []
        scores:         list[float] = []
        hop_profiles:   list[list[float]] = []
        spectral_list:  list[list[float]] = []
        venue_ids:      list[int]   = []
        publisher_ids:  list[int]   = []
        years:          list[float] = []

        for entry in self.context_cache.get(doc_id, [])[:self.max_context_size]:
            tensors = self.context_entry_tensors(center_year, entry)
            # Skip entries that no longer resolve to a valid document row.
            if tensors is None:
                continue
            # The neighbour's text tensors are the heaviest part of the data, 
            # so read them first and skip the rest of the entry when the neighbour 
            # document is missing from this split.
            neighbors.append(tensors["text"])
            edge_types.append(tensors["edge_type"])
            year_deltas.append(tensors["year_delta"])
            scores.append(tensors["score"])
            venue_ids.append(tensors["venue_id"])
            publisher_ids.append(tensors["publisher_id"])
            years.append(tensors["year"])
    
            if self.hop_profile_dim > 0:
                hop_profiles.append(tensors.get("hop_profile", [0.0] * self.hop_profile_dim))
            if self.spectral_dim > 0:
                spectral_list.append(tensors.get("spectral", [0.0] * self.spectral_dim))

        # How many real entries we have; the rest will be zero-padding.
        valid_count = len(neighbors)
        pad_count   = max(0, self.max_context_size - valid_count)

        # Pad all lists to max_context_size so the final tensors have a fixed shape.
        neighbors     += [self.empty_text] * pad_count
        edge_types    += [int(EdgeType.NONE)] * pad_count
        year_deltas   += [0.0] * pad_count
        scores        += [0.0] * pad_count
        venue_ids     += [0]   * pad_count
        publisher_ids += [0]   * pad_count
        years         += [0.0] * pad_count
        if self.hop_profile_dim > 0:
            hop_profiles  += [[0.0] * self.hop_profile_dim] * pad_count
        if self.spectral_dim > 0:
            spectral_list += [[0.0] * self.spectral_dim]    * pad_count

        # Stack neighbour text tensors along the context axis.
        item["context_input_ids"]      = torch.stack([n["input_ids"]      for n in neighbors])
        item["context_attention_mask"] = torch.stack([n["attention_mask"] for n in neighbors])

        # Binary mask: 1 for real neighbours, 0 for padding slots.
        item["context_mask"]          = torch.tensor([1] * valid_count + [0] * pad_count, dtype=torch.long)
        item["context_edge_types"]    = torch.tensor(edge_types,    dtype=torch.long)
        item["context_year_deltas"]   = torch.tensor(year_deltas,   dtype=torch.float32)
        item["context_scores"]        = torch.tensor(scores,        dtype=torch.float32)
        item["context_venue_ids"]     = torch.tensor(venue_ids,     dtype=torch.long)
        item["context_publisher_ids"] = torch.tensor(publisher_ids, dtype=torch.long)
        item["context_years"]         = torch.tensor(years,         dtype=torch.float32)

        if self.hop_profile_dim > 0:
            item["context_hop_profiles"] = torch.tensor(hop_profiles, dtype=torch.float32)
        if self.spectral_dim > 0:
            item["context_spectral"]     = torch.tensor(spectral_list, dtype=torch.float32)

        # Label is optional: unlabeled rows (semi-supervised unlabeled split)
        # omit 'labels' from the dict so the model can skip the loss on them.
        label = row["label"]
        if label is not None and not (isinstance(label, float) and np.isnan(label)):
            item["labels"] = torch.tensor(int(label), dtype=torch.long)

        return item


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 1) -> DataLoader:
    """
    Pinned memory is only enabled when a CUDA device is available, it has no
    effect on CPU-only machines and wastes RAM there.  Worker persistence and
    prefetching are only turned on when worker processes actually exist because
    the settings are meaningless in single-process mode and can cause warnings.

    The prefetch factor controls how many batches each worker prepares ahead of the training loop.
    """
    kwargs: dict[str, Any] = {
        "dataset":     dataset,
        "batch_size":  batch_size,
        "shuffle":     shuffle,
        "num_workers": num_workers,
        "pin_memory":  torch.cuda.is_available()
    }
    if num_workers > 0:
        kwargs |= {"persistent_workers": True, "prefetch_factor": DATALOADER_PREFETCH_FACTOR}
    return DataLoader(**kwargs)
