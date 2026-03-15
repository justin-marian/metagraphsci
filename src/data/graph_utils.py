from pathlib import Path
from typing import Iterable

import polars as pl
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, subgraph as pyg_subgraph

from constants import EdgeType, GraphData


def finalize_graph_data(graph: GraphData) -> GraphData:
    """Attach fast lookup structures to a PyG citation graph."""
    
    # O(1) Graph Lookups
    # PyTorch Geometric represents edges as a [2, num_edges] tensor (`edge_index`). 
    # Finding all neighbors for a node requires scanning this entire tensor, which is 
    # disastrously slow in a Python loop. By precomputing standard adjacency lists 
    # (dictionaries and sets) immediately after loading, we enable O(1) local graph 
    # traversals for the rest of the pipeline.
    node_ids = graph.node_ids.tolist() if hasattr(graph, "node_ids") else list(range(int(graph.num_nodes)))
    graph.node_id_to_idx = {int(node_id): idx for idx, node_id in enumerate(node_ids)}

    out_neighbors: dict[int, set[int]] = {int(n): set() for n in node_ids}
    in_neighbors: dict[int, set[int]] = {int(n): set() for n in node_ids}
    edge_set: set[tuple[int, int]] = set()

    if getattr(graph, "edge_index", None) is not None and graph.edge_index.numel() > 0:
        src, dst = graph.edge_index[0].cpu().tolist(), graph.edge_index[1].cpu().tolist()
        for s_idx, d_idx in zip(src, dst):
            s_id, d_id = int(node_ids[int(s_idx)]), int(node_ids[int(d_idx)])
            out_neighbors[s_id].add(d_id)
            in_neighbors[d_id].add(s_id)
            edge_set.add((s_id, d_id))

    graph.out_neighbors, graph.in_neighbors, graph.edge_set = out_neighbors, in_neighbors, edge_set
    return graph


def load_citation_graph(path: str | Path, source_col: str = "source", target_col: str = "target", node_ids: Iterable[int] | None = None) -> GraphData:
    """Load the directed citation graph as a torch_geometric Data object."""
    
    # ARCHITECTURAL DECISION: Node ID Decoupling
    # Real-world dataset IDs (like arXiv IDs or MAG IDs) are rarely contiguous, 
    # 0-indexed integers. If fed directly to PyG, it allocates massive sparse matrices 
    # leading to Out-Of-Memory errors. We explicitly map physical `doc_id`s to 
    # continuous 0-indexed internal indices to keep tensors compact.
    path = Path(path)
    if path.suffix == ".csv":
        edges = pl.read_csv(path)
    elif path.suffix in {".parquet", ".pq"}: 
        edges = pl.read_parquet(path)
    else: 
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if source_col not in edges.columns or target_col not in edges.columns:
        raise ValueError(f"Citation file must contain '{source_col}' and '{target_col}' columns.")

    # Native Polars casting to integer lists
    src_ids = edges[source_col].cast(pl.Int64).to_list()
    dst_ids = edges[target_col].cast(pl.Int64).to_list()

    all_node_ids = sorted(set(map(int, node_ids or [])) |  set(src_ids) | set(dst_ids))
    node_id_to_idx = {n: i for i, n in enumerate(all_node_ids)}

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = Tensor([[node_id_to_idx[int(s)] for s in src_ids], [node_id_to_idx[int(d)] for d in dst_ids]], dtype=torch.long)

    graph = Data(edge_index=edge_index, num_nodes=len(all_node_ids))
    graph.node_ids = Tensor(all_node_ids, dtype=torch.long)
    return finalize_graph_data(graph)


def subgraph_by_doc_ids(graph: GraphData, doc_ids: Iterable[int]) -> GraphData:
    doc_id_set = set(map(int, doc_ids))
    subset_indices = [graph.node_id_to_idx[n] for n in graph.node_ids.tolist() if int(n) in doc_id_set]
    subset = Tensor(subset_indices, dtype=torch.long)
    edge_index, _ = pyg_subgraph(subset, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes, return_edge_mask=True)
    
    sub = Data(edge_index=edge_index, num_nodes=subset.numel())
    sub.node_ids = graph.node_ids[subset]
    return finalize_graph_data(sub)


def split_graphs(graph: GraphData, train_ids: Iterable[int], val_ids: Iterable[int], test_ids: Iterable[int], mode: str) -> dict[str, GraphData]:
    # Transductive vs. Inductive Evaluation
    # Graph models have two learning environments:
    # 1. Transductive: The full graph structure (including test node edges) is visible during 
    #    training, but test labels are hidden. Useful for single-network scenarios.
    # 2. Inductive: Test nodes and their edges are physically severed from the training graph 
    #    and only reintroduced at evaluation. This proves the model can generalize to unseen data.
    mode = mode.lower()
    train_set, val_set, test_set = set(map(int, train_ids)), set(map(int, val_ids)), set(map(int, test_ids))

    if mode == "transductive":
        return {"pretrain": finalize_graph_data(graph.clone()), 
                "val": finalize_graph_data(graph.clone()), 
                "test": finalize_graph_data(graph.clone())}
    if mode == "inductive":
        return {"pretrain": subgraph_by_doc_ids(graph, train_set), 
                "val": subgraph_by_doc_ids(graph, val_set), 
                "test": subgraph_by_doc_ids(graph, test_set)}
    if mode == "train_plus_eval":
        return {"pretrain": subgraph_by_doc_ids(graph, train_set), 
                "val": subgraph_by_doc_ids(graph, train_set | val_set), 
                "test": subgraph_by_doc_ids(graph, train_set | test_set)}
    
    raise ValueError(f"Unknown graph mode: {mode}")


def build_local_context_map(graph: GraphData) -> dict[int, set[int]]:
    return {int(n): 
        set(graph.out_neighbors.get(int(n), set())) | 
        set(graph.in_neighbors.get(int(n), set())) - {int(n)} 
        for n in graph.node_ids.tolist()}


def edge_type(graph: GraphData, center_id: int, neighbor_id: int) -> EdgeType:
    incoming = (int(neighbor_id), int(center_id)) in graph.edge_set
    outgoing = (int(center_id), int(neighbor_id)) in graph.edge_set
    if incoming and outgoing: 
        return EdgeType.BIDIRECTIONAL
    if incoming:
        return EdgeType.IN
    if outgoing: 
        return EdgeType.OUT
    return EdgeType.NONE


def reciprocity_value(edge_type: EdgeType) -> float:
    return 1.0 if edge_type == EdgeType.BIDIRECTIONAL else 0.0


def overlap_score(local_contexts: dict[int, set[int]], node_id: int, neighbor_id: int) -> float:
    node_ctx, neighbor_ctx = local_contexts.get(node_id, set()), local_contexts.get(neighbor_id, set())
    union = node_ctx | neighbor_ctx
    return len(node_ctx & neighbor_ctx) / len(union) if union else 0.0


def k_hop_profile(graph: GraphData, node_id: int, max_hops: int) -> list[float]:
    """Count the number of nodes at each hop distance around a node using BFS."""
    if max_hops <= 0 or int(node_id) not in graph.node_id_to_idx: 
        return []

    undirected = {n:
        set(graph.out_neighbors.get(n, set())) |
        set(graph.in_neighbors.get(n, set())) 
        for n in graph.node_id_to_idx.keys()}
    visited, frontier, counts = {int(node_id)}, {int(node_id)}, [0.0] * max_hops

    for hop in range(1, max_hops + 1):
        next_frontier = set().union(*(undirected.get(c, set()) for c in frontier)) - visited
        counts[hop - 1] = float(len(next_frontier))
        if not next_frontier: 
            break
        visited |= next_frontier
        frontier = next_frontier

    total = sum(counts)
    return [v / total for v in counts] if total > 0 else counts


def spectral_features(graph: GraphData, node_ids: Iterable[int], spectral_dim: int, enabled: bool = False, max_nodes: int = 5000) -> dict[int, list[float]]:
    # Laplacian Eigenvectors as Positional Encodings
    # Pure Transformers treat inputs as an unordered set. While we explicitly inject 
    # edge directions and year deltas, injecting the eigenvectors of the graph Laplacian 
    # gives the Transformer a sense of "Global Graph Coordinates". 
    # However, Eigen-decomposition is an O(V^3) operation, which is why it is gated by 
    # `max_nodes` to prevent infinite hangs on massive subgraphs.
    node_list = list(map(int, node_ids))
    if spectral_dim <= 0 or not enabled or len(node_list) > max_nodes:
        return {n: [0.0] * spectral_dim for n in node_list}

    subset_indices = [graph.node_id_to_idx[n] for n in node_list if n in graph.node_id_to_idx]
    if not subset_indices: 
        return {n: [0.0] * spectral_dim for n in node_list}

    subset = Tensor(subset_indices, dtype=torch.long)
    edge_index, _ = pyg_subgraph(subset, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes, return_edge_mask=True)
    if subset.numel() == 0: 
        return {n: [0.0] * spectral_dim for n in node_list}

    undirected_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1) if edge_index.numel() > 0 else edge_index
    lap_edge_index, lap_weight = get_laplacian(undirected_edge_index, normalization="sym", num_nodes=subset.numel())
    
    lap = torch.zeros((subset.numel(), subset.numel()), dtype=torch.float32)
    if lap_edge_index.numel() > 0:
        lap[lap_edge_index[0], lap_edge_index[1]] = lap_weight.float()

    _, eigenvectors = torch.linalg.eigh(lap)
    use_dim = min(spectral_dim, max(0, eigenvectors.size(1) - 1))
    features = torch.zeros((subset.numel(), spectral_dim), dtype=torch.float32)
    
    if use_dim > 0:
        # Ignore the first eigenvector (which is a constant for connected components)
        features[:, :use_dim] = eigenvectors[:, 1:1 + use_dim]

    return {
        int(graph.node_ids[int(orig_idx)]): features[row_idx].tolist() 
        for row_idx, orig_idx in enumerate(subset.tolist())
    }
