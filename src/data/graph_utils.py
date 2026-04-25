"""
Graph loading and structural feature helpers for citation graphs.

1.  Loading a citation edge list from disk and building a PyG Data object with ense tensor index remapping.
2.  Attaching fast adjacency dictionaries (out_neighbors, in_neighbors, edge_set) to any PyG Data object so 
    local neighbourhood queries are O(1) per edge instead of O(E).
3.  Extracting train/val/test graph views in transductive, inductive, or mixed modes.
4. Computing structural node features: BFS hop profiles (k hop profile) and Laplacian spectral coordinates (spectral features).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl
import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, subgraph as pyg_subgraph

from constants import EdgeType, GraphData, MAX_NODES_FOR_SPECTRAL


def graph_node_ids(graph: GraphData) -> list[int]:
    """
    Return the external document ids stored in the graph.

    Some graphs carry original document ids; others only expose a node count 
    (in which case dense integer indices are used as surrogates). 

    Normalising that difference here keeps the rest of the
    module independent from how the graph was constructed.
    """
    if hasattr(graph, "node_ids"):
        return [int(nid) for nid in graph.node_ids.tolist()]
    return list(range(int(graph.num_nodes)))


def read_edges(path: Path) -> pl.DataFrame:
    """Load an edge table from a supported citation file format (CSV or Parquet)."""
    if path.suffix == ".csv":
        return pl.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path.suffix!r}")


def build_undirected_neighbors(graph: GraphData) -> dict[int, set[int]]:
    """
    Merge incoming and outgoing adjacency sets for undirected traversals.

    Used by k hop profile, which treats the citation graph as undirected when
    computing BFS hop counts so that both references and reverse-citations
    count as graph distance.
    """
    return {
        node_id: (graph.out_neighbors.get(node_id, set()) | graph.in_neighbors.get(node_id, set()))
        for node_id in graph.node_id_to_idx}


def finalize_graph_data(graph: GraphData) -> GraphData:
    """
    Attach fast adjacency structures to a PyG Data object in place.

    Efficient for tensor operations (scatter, message passing) 
    but slow for repeated local-neighbourhood queries (O(E) scan per node). 
    Three complementary structures:

        node_id_to_idx => {external doc_id => dense tensor index}
        out_neighbors  => {doc_id => set of doc_ids this node cites}
        in_neighbors   => {doc_id => set of doc_ids that cite this node}
        edge_set       => set of (src_doc_id, dst_doc_id) pairs for O(1) lookup

    These are read-only after construction; mutating them without re-calling
    this function will leave them inconsistent with edge_index.
    """
    node_ids = graph_node_ids(graph)
    graph.node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    graph.out_neighbors: dict[int, set[int]] = {nid: set() for nid in node_ids}
    graph.in_neighbors:  dict[int, set[int]] = {nid: set() for nid in node_ids}
    graph.edge_set:      set[tuple[int, int]] = set()

    edge_index = getattr(graph, "edge_index", None)
    if edge_index is not None and edge_index.numel() > 0:
        for src_idx, dst_idx in edge_index.t().cpu().tolist():
            src_id = node_ids[int(src_idx)]
            dst_id = node_ids[int(dst_idx)]
            graph.out_neighbors[src_id].add(dst_id)
            graph.in_neighbors[dst_id].add(src_id)
            graph.edge_set.add((src_id, dst_id))

    return graph


def load_citation_graph(
    path: str | Path,
    source_col: str = "source", target_col: str = "target",
    node_ids: Iterable[int] | None = None) -> GraphData:
    """
    Load a citation graph and remap real document ids to dense tensor indices.

    The edge table is read from disk, source and target columns are extracted,
    and all unique document ids (from both the edge list and the optional
    node_ids iterable) are sorted into a deterministic dense mapping.

    An empty edge_index tensor is used when the edge table is empty so that
    finalize_graph_data still runs correctly and the resulting graph has the correct node count.

    Parameters
    ----------
    path       :    Path to a CSV or Parquet edge file.
    source_col :    Column containing the citing document id.
    target_col :    Column containing the cited document id.
    node_ids   :    Optional iterable of additional node ids to include even if
                    they have no edges (e.g. isolated nodes in the document table).
    """
    edges = read_edges(Path(path))
    if source_col not in edges.columns or target_col not in edges.columns:
        raise ValueError(f"Citation file must contain {source_col!r} and {target_col!r} columns.")

    src_ids = [int(v) for v in edges[source_col].cast(pl.Int64).to_list()]
    dst_ids = [int(v) for v in edges[target_col].cast(pl.Int64).to_list()]

    # Union of edge-list nodes and any caller-supplied ids (e.g. document table).
    all_node_ids   = sorted(set(map(int, node_ids or [])) | set(src_ids) | set(dst_ids))
    node_id_to_idx = {nid: idx for idx, nid in enumerate(all_node_ids)}

    edge_index = (
        torch.empty((2, 0), dtype=torch.long)
        if not src_ids else torch.tensor([
            [node_id_to_idx[s] for s in src_ids],
            [node_id_to_idx[d] for d in dst_ids]
        ], dtype=torch.long))

    graph = Data(edge_index=edge_index, num_nodes=len(all_node_ids))
    graph.node_ids = torch.tensor(all_node_ids, dtype=torch.long)
    return finalize_graph_data(graph)


def subgraph_by_doc_ids(graph: GraphData, doc_ids: Iterable[int]) -> GraphData:
    """
    Extract the induced subgraph over a set of external document ids.

    Only nodes whose ids appear in doc_ids (and whose tensor index exists in node_id_to_idx) are included.
    Edges between nodes outside the subset are removed, and nodes are relabelled to a fresh dense index.
    """
    doc_id_set = {int(did) for did in doc_ids}
    subset = torch.tensor([graph.node_id_to_idx[nid] for nid in graph_node_ids(graph) if nid in doc_id_set], dtype=torch.long)
    edge_index, _ = pyg_subgraph(subset, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes, return_edge_mask=True)
    subgraph = Data(edge_index=edge_index, num_nodes=subset.numel())
    subgraph.node_ids = graph.node_ids[subset]
    return finalize_graph_data(subgraph)


def split_graphs(
    graph: GraphData,
    train_ids: Iterable[int],
    val_ids: Iterable[int],
    test_ids: Iterable[int],
    mode: str) -> dict[str, GraphData]:
    """Build train/val/test graph views according to the requested graph regime.

    Supported modes
    ---------------
    "transductive"    => All three splits share the full graph.  Labels are
                        masked at training time by the model, not here.  Simple
                        and fast; the standard choice for node classification.

    "inductive"       => Each split receives its own subgraph containing only
                        its own nodes.  Stricter: the model never sees test-node
                        embeddings during training.  Produces lower baseline
                        numbers but better reflects production deployment.

    "train_plus_eval" => Training subgraph contains only train nodes.  Val and
                        test subgraphs additionally include training nodes so
                        label propagation and transductive aggregation can still
                        use the training neighbourhood during evaluation.
    """
    mode      = mode.lower()
    train_set = {int(nid) for nid in train_ids}
    val_set   = {int(nid) for nid in val_ids}
    test_set  = {int(nid) for nid in test_ids}

    if mode == "transductive":
        full_graph = finalize_graph_data(graph.clone())
        return {"pretrain": full_graph, "val": full_graph, "test": full_graph}
    if mode == "inductive":
        return {
            "pretrain": subgraph_by_doc_ids(graph, train_set),
            "val":      subgraph_by_doc_ids(graph, val_set),
            "test":     subgraph_by_doc_ids(graph, test_set)}
    if mode == "train_plus_eval":
        return {
            "pretrain": subgraph_by_doc_ids(graph, train_set),
            "val":      subgraph_by_doc_ids(graph, train_set | val_set),
            "test":     subgraph_by_doc_ids(graph, train_set | test_set)}
    raise ValueError(f"Unknown graph mode: {mode!r}")


def build_local_context_map(graph: GraphData) -> dict[int, set[int]]:
    """
    Build the undirected one-hop neighbourhood map with self-loops removed.

    Returns {doc_id => set of neighbour doc_ids} using the union of incoming
    and outgoing edges.  The center node itself is excluded so scoring
    functions never accidentally compare a node to itself.
    """
    return {node_id: (
        graph.out_neighbors.get(node_id, set()) |
        graph.in_neighbors.get(node_id, set())) - {node_id}
    for node_id in graph_node_ids(graph)}


def edge_type(graph: GraphData, center_id: int, neighbor_id: int) -> EdgeType:
    """Classify the directed citation relation between two nodes.

    Checks the pre-built edge_set for both directions in O(1) and returns the
    appropriate EdgeType enum value.  NONE is returned for pairs with no edge.
    """
    center_id, neighbor_id = int(center_id), int(neighbor_id)
    has_incoming = (neighbor_id, center_id) in graph.edge_set  # neighbour cites center
    has_outgoing = (center_id, neighbor_id) in graph.edge_set  # center cites neighbour

    if has_incoming and has_outgoing:
        return EdgeType.BIDIRECTIONAL
    if has_incoming:
        return EdgeType.IN
    if has_outgoing:
        return EdgeType.OUT
    return EdgeType.NONE


def reciprocity_value(etype: EdgeType) -> float:
    """
    Return the scalar reciprocity feature used by the neighbour scoring model.

    Only bidirectional citations receive a non-zero score because mutual
    citations are the strongest indicator of intellectual kinship between two papers.
    """
    return 1.0 if etype == EdgeType.BIDIRECTIONAL else 0.0


def overlap_score(local_contexts: dict[int, set[int]], node_id: int, neighbor_id: int) -> float:
    """
    Compute Jaccard overlap between the one-hop neighbourhoods of two nodes.

    A high overlap means the two papers share many co-citations / co-reference
    targets, which is a strong indicator of topical proximity.  Returns 0.0
    when the union is empty (isolated or degree-0 nodes).
    """
    node_context     = local_contexts.get(int(node_id), set())
    neighbor_context = local_contexts.get(int(neighbor_id), set())
    union = node_context | neighbor_context
    return len(node_context & neighbor_context) / len(union) if union else 0.0


def k_hop_profile(graph: GraphData, node_id: int, max_hops: int) -> list[float]:
    """Compute the normalised BFS hop-count profile for one node.

    The profile is a list of length max_hops where position i contains the
    fraction of reachable nodes first reached at exactly hop i.  Normalisation
    by total reachable nodes makes the feature invariant to graph size.

    An undirected adjacency view is used so that both outgoing references and
    incoming citations contribute to the hop distance.

    Returns an empty list when max_hops ≤ 0 or the node is not in the graph.
    """
    node_id = int(node_id)
    if max_hops <= 0 or node_id not in graph.node_id_to_idx:
        return []

    undirected = build_undirected_neighbors(graph)
    visited    = {node_id}
    frontier   = {node_id}
    counts     = [0.0] * max_hops

    for hop in range(max_hops):
        # Expand one BFS layer; subtract already-visited nodes to keep exact
        # hop distances (a node is counted only at the first hop it is reached).
        next_frontier = set().union(*(undirected.get(node, set()) for node in frontier)) - visited
        counts[hop] = float(len(next_frontier))
        if not next_frontier:
            break  # Graph exhausted before reaching max_hops.
        visited  |= next_frontier
        frontier  = next_frontier

    # Normalise by total reachable nodes so the profile sums to 1.0 (or
    # remains all-zeros for isolated nodes, which is the correct signal).
    total = sum(counts)
    return [c / total for c in counts] if total > 0 else counts


def spectral_features(
    graph: GraphData, node_ids: Iterable[int],
    spectral_dim: int, enabled: bool = False,
    max_nodes: int = MAX_NODES_FOR_SPECTRAL
) -> dict[int, list[float]]:
    """
    Laplacian eigenvector features for a requested node subset.

    The features (also called Laplacian positional encodings) are columns 1…k
    of the eigenvector matrix of the symmetric normalised Laplacian.  Column 0
    (the constant eigenvector) is skipped because it carries no positional
    signal within a connected component.

    Computation is gated by three conditions (all must be met):
        - enabled must be True.
        - spectral_dim must be > 0.
        - The number of requested nodes must not exceed max_nodes

    An empty dict is returned when any condition is not met, so callers receive
    a consistent type regardless of whether spectral features were computed.

    Parameters
    ----------
    graph        : Citation graph with node_ids and adjacency info.
    node_ids     : Nodes for which to return eigenvector rows.
    spectral_dim : Number of eigenvector columns to extract.
    enabled      : Master switch; set False to skip computation entirely.
    max_nodes    : Size cap; eigendecomposition on large graphs is too slow.
    """
    node_list = [int(nid) for nid in node_ids]

    # Fast early exit when any disabling condition is met.
    if not enabled or spectral_dim <= 0 or len(node_list) > max_nodes:
        return {}

    # Filter to nodes that actually appear in the graph's dense index.
    valid = [nid for nid in node_list if nid in graph.node_id_to_idx]
    if not valid:
        return {}

    subset = torch.tensor([graph.node_id_to_idx[nid] for nid in valid], dtype=torch.long)
    edge_index, _ = pyg_subgraph(
        subset, graph.edge_index, relabel_nodes=True,
        num_nodes=graph.num_nodes, return_edge_mask=True)

    # Symmetric normalised Laplacian keeps the eigensystem real-valued and
    # numerically stable for downstream positional features.  Symmetrise the
    # directed subgraph before computing the Laplacian.
    undirected_ei = (
        torch.cat([edge_index, edge_index.flip(0)], dim=1)
        if edge_index.numel() > 0 else edge_index)
    lap_ei, lap_w = get_laplacian(undirected_ei, normalization="sym", num_nodes=subset.numel())

    laplacian = torch.zeros((subset.numel(), subset.numel()), dtype=torch.float32)
    if lap_ei.numel() > 0:
        laplacian[lap_ei[0], lap_ei[1]] = lap_w.float()

    # torch.linalg.eigh is more stable than eig for real symmetric matrices
    # and returns eigenvalues in ascending order, so column 0 is always the
    # all-constant eigenvector and column 1 is the Fiedler vector.
    _, eigenvectors = torch.linalg.eigh(laplacian)

    # Skip column 0 (constant); take up to spectral_dim remaining columns.
    usable_dim = min(spectral_dim, max(0, eigenvectors.size(1) - 1))
    features   = torch.zeros((subset.numel(), spectral_dim), dtype=torch.float32)
    if usable_dim > 0:
        features[:, :usable_dim] = eigenvectors[:, 1 : 1 + usable_dim]

    return {
        int(graph.node_ids[int(original_idx)]): features[row_idx].tolist()
        for row_idx, original_idx in enumerate(subset.tolist())
    }
