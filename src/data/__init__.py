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

from .context_caching import build_neighbor_cache, load_neighbor_cache, save_neighbor_cache
from .dataset import MultiScaleDocumentDataset, NeighborCache, build_loader, create_tokenizer
from .graph_utils import load_citation_graph, split_graphs
from .tabular_utils import create_encoders, create_low_label_split, load_documents, prepare_documents, split_documents

__all__ = [
    "MultiScaleDocumentDataset", "NeighborCache",
    "build_loader", "build_neighbor_cache",
    "create_encoders", "create_low_label_split", "create_tokenizer",
    "load_citation_graph", "load_documents", "load_neighbor_cache",
    "prepare_documents", "save_neighbor_cache",
    "split_documents", "split_graphs"
]
