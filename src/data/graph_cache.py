"""Disk caching for citation graphs and train/val/test split views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import polars as pl
import torch
from loguru import logger
from torch_geometric.data import Data

from .cache_utils import citation_edges_fingerprint, docs_fingerprint, metadata_matches
from .constants import GraphData
from .graph_utils import finalize_graph_data, load_citation_graph, split_graphs

SplitGraphs: TypeAlias = dict[str, GraphData]

COMPATIBILITY_KEYS = (
    "seed", "graph_mode", "split_strategy", "test_size", "val_size",
    "source_col", "target_col", "citations_path", "documents_path",
    "docs_fingerprint", "citation_edges_fingerprint")


def compute_graph_metadata(data_cfg: dict[str, Any], documents: pl.DataFrame, seed: int, citations: pl.DataFrame | None = None) -> dict[str, Any]:
    """Fingerprint the inputs that determine graph topology and split layout."""
    citation_source: pl.DataFrame | str | Path = citations if citations is not None else data_cfg["citations"]
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
        "citation_edges_fingerprint": citation_edges_fingerprint(
            citation_source, str(data_cfg["source_col"]), str(data_cfg["target_col"]))}


def graph_is_compatible(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return True when a saved graph cache still matches current settings."""
    return metadata_matches("graph", metadata, expected, COMPATIBILITY_KEYS)


def graph_to_dict(graph: GraphData) -> dict[str, torch.Tensor]:
    """Serialise the primitive tensors that fully describe a graph view."""
    return {
        "edge_index": graph.edge_index.detach().cpu(),
        "node_ids": graph.node_ids.detach().cpu(),
        "num_nodes": torch.tensor(int(graph.num_nodes), dtype=torch.long)}


def dict_to_graph(payload: dict[str, torch.Tensor]) -> GraphData:
    """Rebuild a finalised PyG graph with adjacency caches from saved tensors."""
    graph = Data(edge_index=payload["edge_index"].long(), num_nodes=int(payload["num_nodes"].item()))
    graph.node_ids = payload["node_ids"].long()
    return finalize_graph_data(graph)


def save_graph_cache(full_graph: GraphData, splits: SplitGraphs, path: str | Path, metadata: dict[str, Any]) -> None:
    """Persist the full graph plus its split views and build metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "metadata": metadata,
        "full": graph_to_dict(full_graph),
        "splits": {name: graph_to_dict(graph) for name, graph in splits.items()}}, output_path)

    logger.info("Saved graph cache: {} split views -> {}", len(splits), output_path)


def load_graph_cache(path: str | Path) -> tuple[GraphData, SplitGraphs, dict[str, Any]]:
    """Load the full graph and its split views from disk."""
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Graph cache not found: {cache_path}")

    payload = torch.load(cache_path, weights_only=False)
    full_graph = dict_to_graph(payload["full"])
    splits = {name: dict_to_graph(data) for name, data in payload["splits"].items()}

    logger.info("Loaded graph cache: {}", cache_path)
    return full_graph, splits, payload["metadata"]


def build_graph_cache(
    data_cfg: dict[str, Any], documents: pl.DataFrame,
    train_ids: list[int], val_ids: list[int], test_ids: list[int]) -> tuple[GraphData, SplitGraphs]:
    """Materialise the citation graph and its train/val/test views."""
    full_graph = load_citation_graph(
        data_cfg["citations"], source_col=data_cfg["source_col"],
        target_col=data_cfg["target_col"], node_ids=documents["doc_id"].to_list())
    splits = split_graphs(full_graph, train_ids, val_ids, test_ids, mode=data_cfg["graph_mode"])
    return full_graph, splits
