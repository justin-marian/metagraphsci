"""
Disk caching for the citation graph and its train/val/test splits.

The full edge list and three split views are deterministic functions of
(documents, citations, split parameters, seed, graph_mode), so caching the
materialised PyG `Data` objects skips the edge-table read, the dense index
remap, and the per-split `subgraph_by_doc_ids` traversal on every run.

PyG `Data` is not pickle-serialised here on purpose: only the primitive
tensors (`edge_index`, `node_ids`) are written, and `finalize_graph_data`
rebuilds the adjacency dictionaries on load so the cache stays portable
across PyG versions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .cache_paths import docs_fingerprint
from .constants import GraphData
from .graph_utils import finalize_graph_data, load_citation_graph, split_graphs
from torch_geometric.data import Data
import polars as pl


SplitGraphs = dict[str, GraphData]


def compute_graph_metadata(
    data_cfg: dict[str, Any], documents: pl.DataFrame, seed: int
) -> dict[str, Any]:
    """Fingerprint the inputs that determine graph topology and split layout."""
    return {
        "seed": int(seed),
        "graph_mode": str(data_cfg["graph_mode"]),
        "split_strategy": str(data_cfg["split_strategy"]),
        "test_size": float(data_cfg["test_size"]),
        "val_size": float(data_cfg["val_size"]),
        "source_col": str(data_cfg["source_col"]),
        "target_col": str(data_cfg["target_col"]),
        "citations_path": str(data_cfg["citations"]),
        "documents_path": str(data_cfg["documents"]),
        "docs_fingerprint": docs_fingerprint(documents),
    }


def graph_is_compatible(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Decide whether a saved graph cache still matches current settings."""
    keys = (
        "seed", "graph_mode", "split_strategy", "test_size", "val_size",
        "source_col", "target_col", "citations_path", "documents_path",
        "docs_fingerprint",
    )
    return all(metadata.get(k) == expected[k] for k in keys)


def graph_to_dict(graph: GraphData) -> dict[str, torch.Tensor]:
    """Serialise the primitive tensors that fully describe a graph view."""
    return {
        "edge_index": graph.edge_index.detach().cpu(),
        "node_ids": graph.node_ids.detach().cpu(),
        "num_nodes": torch.tensor(int(graph.num_nodes), dtype=torch.long),
    }


def dict_to_graph(payload: dict[str, torch.Tensor]) -> GraphData:
    """Rebuild a finalised PyG graph (with adjacency caches) from saved tensors."""
    graph = Data(
        edge_index=payload["edge_index"].long(),
        num_nodes=int(payload["num_nodes"].item()),
    )
    graph.node_ids = payload["node_ids"].long()
    return finalize_graph_data(graph)


def save_graph_cache(
    full_graph: GraphData, splits: SplitGraphs,
    path: str | Path, metadata: dict[str, Any],
) -> None:
    """Persist the full graph plus its split views and build metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metadata": metadata,
        "full": graph_to_dict(full_graph),
        "splits": {name: graph_to_dict(g) for name, g in splits.items()},
    }, output_path)


def load_graph_cache(path: str | Path) -> tuple[GraphData, SplitGraphs, dict[str, Any]]:
    """Load the full graph and its split views from disk."""
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Graph cache not found: {cache_path}")
    payload = torch.load(cache_path, weights_only=False)
    full_graph = dict_to_graph(payload["full"])
    splits = {name: dict_to_graph(d) for name, d in payload["splits"].items()}
    return full_graph, splits, payload["metadata"]


def build_graph_cache(
    data_cfg: dict[str, Any], documents: pl.DataFrame,
    train_ids: list[int], val_ids: list[int], test_ids: list[int],
) -> tuple[GraphData, SplitGraphs]:
    """Materialise the citation graph and its train/val/test views."""
    full_graph = load_citation_graph(
        data_cfg["citations"],
        source_col=data_cfg["source_col"], target_col=data_cfg["target_col"],
        node_ids=documents["doc_id"].to_list(),
    )
    splits = split_graphs(full_graph, train_ids, val_ids, test_ids, mode=data_cfg["graph_mode"])
    return full_graph, splits
