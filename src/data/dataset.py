from typing import Any
import numpy as np
import polars as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from constants import UNKNOWN_TOKEN, EdgeType, NeighborCache


def create_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Instantiate the tokenizer used by the shared text encoder."""
    return AutoTokenizer.from_pretrained(model_name)


class MultiScaleDocumentDataset(Dataset):
    """Return one fully prepared multimodal sample per item."""

    def __init__(
        self, documents: pl.DataFrame, tokenizer: PreTrainedTokenizerBase,
        venue_encoder: dict[str, int], publisher_encoder: dict[str, int], author_encoder: dict[str, int],
        max_seq_length: int, max_context_size: int, max_authors: int,
        context_documents: pl.DataFrame | None = None, context_cache: NeighborCache | None = None,
        cache_text: bool = True, pretokenize_context: bool = False,
        hop_profile_dim: int = 2, spectral_dim: int = 0
    ) -> None:
        # Ego-centric Graph Flattening
        # Instead of passing the whole PyG object and using complex message-passing
        # loaders, ingest a pre-computed `NeighborCache`. This flattens the graph into 
        # bounded, ego-centric sequences. It allows us to use a standard DataLoader 
        # and treat the citation neighborhood exactly like a text sequence for the Transformer.
        self.documents = documents.clone()
        self.context = context_documents.clone() if context_documents is not None else documents.clone()

        self.tokenizer = tokenizer
        self.venue_encoder = venue_encoder
        self.publisher_encoder = publisher_encoder
        self.author_encoder = author_encoder

        self.max_seq_length = int(max_seq_length)
        self.max_context_size = int(max_context_size)
        self.max_authors = int(max_authors)
        self.hop_profile_dim = int(hop_profile_dim)
        self.spectral_dim = int(spectral_dim)
        
        # Soft Year Scaling Parameters
        self.reference_year = 2000.0
        self.year_scale = 26.0

        # RAM vs. CPU Compute Trade-off
        # Tokenizing thousands of abstracts during the `__getitem__` call severely bottlenecks 
        # the GPU (as it waits for the CPU). By offering a `pretokenize_context` toggle, 
        # can pay the memory cost upfront to cache the subword tensors, drastically 
        # speeding up training throughput for datasets that fit in RAM.
        self.cache_text = bool(cache_text)
        self.context_cache = context_cache or {}
        
        # Polars optimization: iterate rows as dicts for O(1) lookups during __getitem__
        self.doc_lookup = {int(row["doc_id"]): row for row in self.context.iter_rows(named=True)}
        self.text_cache: dict[int, dict[str, Tensor]] = {}

        if self.cache_text and pretokenize_context:
            for doc_id, row in self.doc_lookup.items():
                self.text_cache[doc_id] = self.tokenize_encode(title=str(row["title"]), abstract=str(row["abstract"]))

    def __len__(self) -> int:
        return self.documents.height

    def tokenize_encode(self, title: str, abstract: str) -> dict[str, Tensor]:
        encoded = self.tokenizer(title, abstract, max_length=self.max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": encoded["input_ids"].squeeze(0), "attention_mask": encoded["attention_mask"].squeeze(0)}

    def text_tokenized(self, doc_id: int, title: str, abstract: str) -> dict[str, Tensor]:
        if self.cache_text and doc_id in self.text_cache:
            return self.text_cache[doc_id]
        tokenized = self.tokenize_encode(title=title, abstract=abstract)
        if self.cache_text:
            self.text_cache[doc_id] = tokenized
        return tokenized

    def parse_year(self, year_val: Any, default: float = 2000.0) -> float:
        """Safely parse missing/NaN years without imposing artificial boundaries."""
        try:
            if year_val is None or (isinstance(year_val, float) and np.isnan(year_val)):
                return default
            return float(year_val)
        except (ValueError, TypeError):
            return default

    def normalize_year(self, year: float) -> float:
        """Scale absolute year to a stable continuous representation."""
        # Papers before 2000 will naturally become negative values, which the 
        # model can easily learn to interpret as historical context.
        return (year - self.reference_year) / self.year_scale

    def metadata_struct(self, row: dict[str, Any]) -> dict[str, Tensor]:
        # Fixed-Size Metadata Tensors & Normalization
        # 1. Authors: Papers have wildly varying author counts. Truncating/padding to `max_authors` 
        #    ensures the tensor shapes remain static, avoiding ragged batches.
        # 2. Year: Neural networks struggle to learn from raw scalar years (e.g., 2023). 
        #    We center the data around the year 2000 and divide by 26 to compress the 
        #    variance, providing a smooth, zero-centered distribution.
        authors = row.get("authors", [])
        authors = authors[:self.max_authors] if isinstance(authors, list) else []
        author_ids = [self.author_encoder.get(a, 0) for a in authors]
        author_ids += [0] * max(0, self.max_authors - len(author_ids)) 
        
        parsed_year = self.parse_year(row.get("year"))
        normalized_year = self.normalize_year(parsed_year)
        
        return {
            "venue_ids": torch.tensor(self.venue_encoder.get(str(row.get("venue", UNKNOWN_TOKEN)), 0), dtype=torch.long),
            "publisher_ids": torch.tensor(self.publisher_encoder.get(str(row.get("publisher", UNKNOWN_TOKEN)), 0), dtype=torch.long),
            "author_ids": torch.tensor(author_ids[:self.max_authors], dtype=torch.long),
            "years": torch.tensor([normalized_year], dtype=torch.float32)
        }

    def empty_neighbor_text(self) -> dict[str, Tensor]:
        return {"input_ids": torch.zeros(self.max_seq_length, dtype=torch.long), "attention_mask": torch.zeros(self.max_seq_length, dtype=torch.long)}

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Retrieve a single fully contextualized multimodal sample."""
        row = self.documents.row(idx, named=True)
        doc_id = int(row["doc_id"])
        
        text_features = self.text_tokenized(doc_id=doc_id, title=str(row.get("title", "")), abstract=str(row.get("abstract", "")))
        item: dict[str, Tensor] = {"doc_id": torch.tensor(doc_id, dtype=torch.long), **text_features, **self.metadata_struct(row)}

        # Parse anchor document's year safely for accurate delta calculations
        center_year = self.parse_year(row.get("year"))
        
        entries = self.context_cache.get(doc_id, [])[:self.max_context_size]
        
        neighbors, edge_types, year_deltas, scores = [], [], [], []
        hop_profiles, spectral_features = [], []
        neighbor_venue_ids, neighbor_publisher_ids, neighbor_years = [], [], []

        for entry in entries:
            neighbor_id = int(entry["doc_id"])
            if neighbor_id not in self.doc_lookup:
                continue

            neighbor_row = self.doc_lookup[neighbor_id]
            neighbors.append(self.text_tokenized(doc_id=neighbor_id, title=str(neighbor_row.get("title", "")), abstract=str(neighbor_row.get("abstract", ""))))
            
            # Robust Year and Delta processing
            # We preserve exact relative distance. A 10-year gap behaves identically 
            # whether it is between 2010 and 2000, or 1970 and 1960.
            neighbor_year = self.parse_year(neighbor_row.get("year"), default=center_year)
            
            edge_types.append(int(entry.get("edge_type", int(EdgeType.NONE))))
            year_deltas.append((neighbor_year - center_year) / self.year_scale)

            scores.append(float(entry.get("score", 0.0)))
            hop_profiles.append([float(v) for v in entry.get("hop_profile", [])][:self.hop_profile_dim])
            spectral_features.append([float(v) for v in entry.get("spectral", [])][:self.spectral_dim])

            neighbor_venue_ids.append(self.venue_encoder.get(str(neighbor_row.get("venue", UNKNOWN_TOKEN)), 0))
            neighbor_publisher_ids.append(self.publisher_encoder.get(str(neighbor_row.get("publisher", UNKNOWN_TOKEN)), 0))
            neighbor_years.append(self.normalize_year(neighbor_year))

        # Deterministic Padding and Graph Masking
        # Transformer architectures demand uniform matrix sizes. If a document has fewer 
        # than `max_context_size` citations, pad the structural inputs with zeros and the textual 
        # inputs with empty embeddings. Most importantly, the `context_mask` acts as the attention 
        # mask at the GNN level. It guarantees the model will mathematically ignore the padded 
        # nodes, preventing zero-vectors from poisoning the attention scores.
        valid_count = len(neighbors)
        pad_count = max(0, self.max_context_size - valid_count)
        
        if pad_count > 0:
            neighbors.extend([self.empty_neighbor_text() for _ in range(pad_count)])
            edge_types.extend([int(EdgeType.NONE)] * pad_count)
            year_deltas.extend([0.0] * pad_count)
            scores.extend([0.0] * pad_count)
            hop_profiles.extend([[0.0] * self.hop_profile_dim for _ in range(pad_count)])
            spectral_features.extend([[0.0] * self.spectral_dim for _ in range(pad_count)])
            neighbor_venue_ids.extend([0] * pad_count)
            neighbor_publisher_ids.extend([0] * pad_count)
            neighbor_years.extend([0.0] * pad_count)

        item["context_input_ids"] = torch.stack([n["input_ids"] for n in neighbors])
        item["context_attention_mask"] = torch.stack([n["attention_mask"] for n in neighbors])
        item["context_mask"] = torch.tensor([1] * valid_count + [0] * pad_count, dtype=torch.long)
        
        item["context_edge_types"] = torch.tensor(edge_types, dtype=torch.long)
        item["context_year_deltas"] = torch.tensor(year_deltas, dtype=torch.float32)
        item["context_scores"] = torch.tensor(scores, dtype=torch.float32)
        item["context_hop_profiles"] = torch.tensor(hop_profiles, dtype=torch.float32) if self.hop_profile_dim > 0 else torch.zeros((self.max_context_size, 0), dtype=torch.float32)
        item["context_spectral"] = torch.tensor(spectral_features, dtype=torch.float32) if self.spectral_dim > 0 else torch.zeros((self.max_context_size, 0), dtype=torch.float32)
        item["context_venue_ids"] = torch.tensor(neighbor_venue_ids, dtype=torch.long)
        item["context_publisher_ids"] = torch.tensor(neighbor_publisher_ids, dtype=torch.long)
        item["context_years"] = torch.tensor(neighbor_years, dtype=torch.float32)

        if "label" in row and row["label"] is not None and not (isinstance(row["label"], float) and np.isnan(row["label"])):
            item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
            
        return item


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    """Create a DataLoader with sensible defaults for this project."""
    kwargs: dict[str, Any] = {
        "dataset": dataset, "batch_size": batch_size,
        "shuffle": shuffle, "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available()
    }
    if num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(**kwargs)
