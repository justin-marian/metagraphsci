"""
- dataset preparation
- graph loading and splitting
- citation context caching
- tokenizer + dataloader construction
- benchmark downloading utilities
"""

from __future__ import annotations

from .dataset import MultiScaleDocumentDataset, NeighborCache, create_tokenizer, build_loader
from .context_caching import build_neighbor_cache, load_neighbor_cache, save_neighbor_cache
from .downloaders import download_planetoid_dataset, download_ogbn_arxiv, download_forc2025
from .tabular_utils import load_documents, prepare_documents, create_encoders, split_documents, create_low_label_split
from .graph_utils import load_citation_graph, split_graphs


__all__ = [
    # Dataset
    "MultiScaleDocumentDataset",
    "NeighborCache",
    "create_tokenizer",
    "build_loader",
    # Context caching
    "build_neighbor_cache",
    "load_neighbor_cache",
    "save_neighbor_cache",
    # Download utilities
    "download_planetoid_dataset",
    "download_ogbn_arxiv",
    "download_forc2025",
    # Tabular preprocessing
    "load_documents",
    "prepare_documents",
    "create_encoders",
    "split_documents",
    "create_low_label_split",
    # Graph utilities
    "load_citation_graph",
    "split_graphs"
]
