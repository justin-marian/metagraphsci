"""
Graph loading and structural feature helpers for citation graphs.

The module keeps only helpers used by the cache, dataset, and graph-splitting
pipeline:

1. Load citation edges from CSV/Parquet and build a PyG Data object with dense remapping.
2. Attach O(1) adjacency dictionaries to PyG graphs for local neighbourhood queries.
3. Extract train/val/test graph views in transductive, inductive, or train-plus-eval modes.
4. Compute optional Laplacian spectral coordinates for cached neighbour records.

BFS hop profiling is precomputed in context_caching.py, where the undirected
adjacency is built once and shared across chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl
import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, subgraph as pyg_subgraph

from .constants import EdgeType, GraphData, MAX_NODES_FOR_SPECTRAL

__all__ = [
    "graph_node_ids", "read_edges", "build_undirected_neighbors",
    "finalize_graph_data", "load_citation_graph", "subgraph_by_doc_ids",
    "split_graphs", "build_local_context_map", "edge_type", "spectral_features"]


def graph_node_ids(graph: GraphData) -> list[int]:
    """Return external document ids, falling back to dense indices when absent."""
    if hasattr(graph, "node_ids"):
        return [int(node_id) for node_id in graph.node_ids.tolist()]
    return list(range(int(graph.num_nodes)))


def read_edges(path: Path) -> pl.DataFrame:
    """Load an edge table from a supported citation file format."""
    if path.suffix == ".csv":
        return pl.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path.suffix!r}")


def build_undirected_neighbors(graph: GraphData) -> dict[int, set[int]]:
    """Merge incoming and outgoing adjacency sets for undirected traversal."""
    return {
        node_id: graph.out_neighbors.get(node_id, set()) | graph.in_neighbors.get(node_id, set())
        for node_id in graph.node_id_to_idx}


def finalize_graph_data(graph: GraphData) -> GraphData:
    """
    Attach fast adjacency structures to a PyG Data object in place.

    Added attributes:
    - node_id_to_idx: external doc_id -> dense tensor index
    - out_neighbors: doc_id -> cited doc_ids
    - in_neighbors: doc_id -> citing doc_ids
    - edge_set: set of (source_doc_id, target_doc_id)
    """
    node_ids = graph_node_ids(graph)
    graph.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    graph.out_neighbors = {node_id: set() for node_id in node_ids}
    graph.in_neighbors = {node_id: set() for node_id in node_ids}
    graph.edge_set = set()

    edge_index = getattr(graph, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return graph

    for src_idx, dst_idx in edge_index.t().cpu().tolist():
        src_id = node_ids[int(src_idx)]
        dst_id = node_ids[int(dst_idx)]
        graph.out_neighbors[src_id].add(dst_id)
        graph.in_neighbors[dst_id].add(src_id)
        graph.edge_set.add((src_id, dst_id))

    return graph


def load_citation_graph(path: str | Path, source_col: str = "source", target_col: str = "target", node_ids: Iterable[int] | None = None) -> GraphData:
    """Load citation edges and remap external document ids to dense tensor indices."""
    edges = read_edges(Path(path))
    if source_col not in edges.columns or target_col not in edges.columns:
        raise ValueError(f"Citation file must contain {source_col!r} and {target_col!r} columns.")

    src_ids = [int(value) for value in edges[source_col].cast(pl.Int64).to_list()]
    dst_ids = [int(value) for value in edges[target_col].cast(pl.Int64).to_list()]
    all_node_ids = sorted(set(map(int, node_ids or [])) | set(src_ids) | set(dst_ids))
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}

    edge_index = (
        torch.empty((2, 0), dtype=torch.long) if not src_ids
        else torch.tensor(
            [[node_id_to_idx[src] for src in src_ids],
            [node_id_to_idx[dst] for dst in dst_ids]], dtype=torch.long))

    graph = Data(edge_index=edge_index, num_nodes=len(all_node_ids))
    graph.node_ids = torch.tensor(all_node_ids, dtype=torch.long)
    return finalize_graph_data(graph)


def subgraph_by_doc_ids(graph: GraphData, doc_ids: Iterable[int]) -> GraphData:
    """Extract the induced subgraph over a set of external document ids."""
    doc_id_set = {int(doc_id) for doc_id in doc_ids}
    subset = torch.tensor(
        [graph.node_id_to_idx[node_id] for node_id in graph_node_ids(graph) if node_id in doc_id_set],
        dtype=torch.long)

    edge_index, _ = pyg_subgraph(
        subset, graph.edge_index, relabel_nodes=True,
        num_nodes=graph.num_nodes, return_edge_mask=True)

    subgraph = Data(edge_index=edge_index, num_nodes=subset.numel())
    subgraph.node_ids = graph.node_ids[subset]
    return finalize_graph_data(subgraph)


def split_graphs(graph: GraphData, train_ids: Iterable[int], val_ids: Iterable[int], test_ids: Iterable[int], mode: str) -> dict[str, GraphData]:
    """Build train/val/test graph views for the requested graph regime."""
    mode = mode.lower()
    train_set = {int(node_id) for node_id in train_ids}
    val_set = {int(node_id) for node_id in val_ids}
    test_set = {int(node_id) for node_id in test_ids}

    if mode == "transductive":
        full_graph = finalize_graph_data(graph.clone())
        return {"pretrain": full_graph, "val": full_graph, "test": full_graph}

    if mode == "inductive":
        return {
            "pretrain": subgraph_by_doc_ids(graph, train_set),
            "val": subgraph_by_doc_ids(graph, val_set),
            "test": subgraph_by_doc_ids(graph, test_set)}

    if mode == "train_plus_eval":
        return {
            "pretrain": subgraph_by_doc_ids(graph, train_set),
            "val": subgraph_by_doc_ids(graph, train_set | val_set),
            "test": subgraph_by_doc_ids(graph, train_set | test_set)}

    raise ValueError(f"Unknown graph mode: {mode!r}")


def build_local_context_map(graph: GraphData) -> dict[int, set[int]]:
    """Build undirected one-hop neighbourhoods with self-loops removed."""
    return {
        node_id: (graph.out_neighbors.get(node_id, set()) | graph.in_neighbors.get(node_id, set())) - {node_id}
        for node_id in graph_node_ids(graph)}


def edge_type(edge_set: set[tuple[int, int]], center_id: int, neighbor_id: int) -> EdgeType:
    """Classify the directed citation relation between two document ids."""
    center_id = int(center_id)
    neighbor_id = int(neighbor_id)
    has_incoming = (neighbor_id, center_id) in edge_set
    has_outgoing = (center_id, neighbor_id) in edge_set

    if has_incoming and has_outgoing:
        return EdgeType.BIDIRECTIONAL
    if has_incoming:
        return EdgeType.IN
    if has_outgoing:
        return EdgeType.OUT
    return EdgeType.NONE


def spectral_features(
    graph: GraphData, node_ids: Iterable[int], spectral_dim: int, enabled: bool = False,
    max_nodes: int = MAX_NODES_FOR_SPECTRAL) -> dict[int, list[float]]:
    """Return Laplacian positional encodings for a requested node subset."""
    node_list = [int(node_id) for node_id in node_ids]
    if not enabled or spectral_dim <= 0 or len(node_list) > max_nodes:
        return {}

    valid = [node_id for node_id in node_list if node_id in graph.node_id_to_idx]
    if not valid:
        return {}

    subset = torch.tensor([graph.node_id_to_idx[node_id] for node_id in valid], dtype=torch.long)
    edge_index, _ = pyg_subgraph(
        subset, graph.edge_index, relabel_nodes=True,
        num_nodes=graph.num_nodes, return_edge_mask=True)

    undirected_edge_index = (
        torch.cat([edge_index, edge_index.flip(0)], dim=1)
        if edge_index.numel() > 0 else edge_index)
    laplacian_index, laplacian_weight = get_laplacian(
        undirected_edge_index, normalization="sym", num_nodes=subset.numel())

    laplacian = torch.zeros((subset.numel(), subset.numel()), dtype=torch.float32)
    if laplacian_index.numel() > 0:
        laplacian[laplacian_index[0], laplacian_index[1]] = laplacian_weight.float()

    _, eigenvectors = torch.linalg.eigh(laplacian)
    usable_dim = min(spectral_dim, max(0, eigenvectors.size(1) - 1))

    features = torch.zeros((subset.numel(), spectral_dim), dtype=torch.float32)
    if usable_dim > 0:
        features[:, :usable_dim] = eigenvectors[:, 1:1 + usable_dim]

    return {
        int(graph.node_ids[int(original_idx)]): features[row_idx].tolist()
        for row_idx, original_idx in enumerate(subset.tolist())}
