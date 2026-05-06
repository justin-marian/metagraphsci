"""
Exposes the end-to-end data pipeline used by training scripts and dataset build jobs:

Tabular layer
-------------
load_documents              - Read and normalise a document CSV/Parquet file.
prepare_documents           - Normalise an in-memory DataFrame to the pipeline schema.
create_encoders             - Build venue / publisher / author vocabulary dicts.
split_documents             - Time-based or stratified train/val/test split.
create_low_label_split      - Semi-supervised labeled / unlabeled partition.

Graph layer
-----------
load_citation_graph         - Load an edge list and build a PyG Data object.
split_graphs                - Create train/val/test graph views (transductive / inductive).

Context caching layer
---------------------
build_neighbor_cache        - Precompute ranked citation-context entries for training.
save_neighbor_cache         - Persist the cache as JSON.
load_neighbor_cache         - Reload the cache with backward-compatible format detection.

Dataset / DataLoader layer
--------------------------
MultiScaleDocumentDataset   - PyTorch Dataset wrapping documents + citation context.
NeighborCache               - Type alias for the precomputed context dict.
create_tokenizer            - Instantiate the shared HuggingFace tokenizer.
build_loader                - Create a DataLoader with project defaults.
"""

from __future__ import annotations

from .cache_paths import cache_root, caching_enabled, docs_fingerprint
from .context_caching import build_neighbor_cache, load_neighbor_cache, save_neighbor_cache
from .dataset import MultiScaleDocumentDataset, NeighborCache, build_loader, create_tokenizer
from .embedding_cache import (
    build_embedding_cache, compute_embedding_metadata, embedding_is_compatible,
    load_embedding_cache, save_embedding_cache)
from .encoder_cache import (
    build_encoder_cache, compute_encoder_metadata, encoder_is_compatible,
    load_encoder_cache, save_encoder_cache)
from .graph_cache import (
    build_graph_cache, compute_graph_metadata, graph_is_compatible,
    load_graph_cache, save_graph_cache)
from .graph_utils import load_citation_graph, split_graphs
from .neighbor_embedding_cache import NeighborEmbeddingCache
from .tabular_utils import create_encoders, create_low_label_split, load_documents, prepare_documents, split_documents
from .tokenization_cache import (
    build_tokenization_cache, compute_tokenization_metadata,
    load_tokenization_cache, save_tokenization_cache, tokenization_is_compatible)

__all__ = [
    "MultiScaleDocumentDataset", "NeighborCache", "NeighborEmbeddingCache",
    "build_embedding_cache", "build_encoder_cache", "build_graph_cache",
    "build_loader", "build_neighbor_cache", "build_tokenization_cache",
    "cache_root", "caching_enabled",
    "compute_embedding_metadata", "compute_encoder_metadata",
    "compute_graph_metadata", "compute_tokenization_metadata",
    "create_encoders", "create_low_label_split", "create_tokenizer",
    "docs_fingerprint",
    "embedding_is_compatible", "encoder_is_compatible", "graph_is_compatible",
    "load_citation_graph", "load_documents", "load_embedding_cache",
    "load_encoder_cache", "load_graph_cache", "load_neighbor_cache",
    "load_tokenization_cache",
    "prepare_documents",
    "save_embedding_cache", "save_encoder_cache", "save_graph_cache",
    "save_neighbor_cache", "save_tokenization_cache",
    "split_documents", "split_graphs", "tokenization_is_compatible"
]
