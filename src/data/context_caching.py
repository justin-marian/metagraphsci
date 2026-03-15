import json
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import polars as pl
from torch_geometric.utils import degree

from constants import EdgeType, NeighborCache, GraphData
from graph_utils import (
    build_local_context_map, edge_type, k_hop_profile,
    overlap_score, reciprocity_value, spectral_features)
from tabular_utils import build_year_lookup


def local_relevance_func(
    graph: GraphData, node_ids: Iterable[int], documents: pl.DataFrame,
    connectivity_weight: float, temporal_weight: float,
    reciprocity_weight: float, overlap_weight: float) -> dict[int, dict[int, float]]:
    """Rank candidate citations with local structural and temporal relevance."""
    
    # Heuristic Graph Sparsification
    # Standard GNNs blindly aggregate all neighbors, which introduces massive noise 
    # (e.g., a paper citing 100 barely-related background papers). Instead of expensive 
    # random walks or PageRank, use a deterministic, multi-factor heuristic to score 
    # and rank neighbors. This forces the model to attend only to the most semantically 
    # and structurally vital citations.
    year_lookup = build_year_lookup(documents)
    local_contexts = build_local_context_map(graph)

    # DESIGN DECISION: Linear Degree Normalization
    # Since explicitly filter out hub nodes (> 50 citations) below, the remaining
    # nodes all have a "low" number of citations. Log-scaling is no longer needed to 
    # dampen extreme outliers. We use raw linear scaling so the model can distinctly
    # differentiate between a paper with 2 citations and one with 45.
    in_deg = degree(graph.edge_index[1], num_nodes=graph.num_nodes).cpu().tolist()
    out_deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes).cpu().tolist()
    in_map = {int(graph.node_ids[i]): float(v) for i, v in enumerate(in_deg)}
    out_map = {int(graph.node_ids[i]): float(v) for i, v in enumerate(out_deg)}

    max_degree = float(max(max(in_map.values(), default=1.0), max(out_map.values(), default=1.0), 1.0))
    relevance: dict[int, dict[int, float]] = {}
    
    for node_id in map(int, node_ids):
        # ! Eliminate the cases with more than 50 citations (Anchor Hubs)
        # Prevents central hub nodes from poisoning the batch context
        if in_map.get(node_id, 0.0) > 50.0:
            relevance[node_id] = {}
            continue

        candidates = local_contexts.get(node_id, set())
        if not candidates:
            relevance[node_id] = {}
            continue

        node_year = year_lookup.get(node_id)
        node_scores: dict[int, float] = {}

        # Multi-Relational Scoring
        # Synthesize 4 distinct graph dynamics into a single relevance score:
        # 1. Connectivity: Is this neighbor globally important?
        # 2. Temporal: Was this published recently relative to the anchor paper? (Exponential decay)
        # 3. Reciprocity: Do these papers cite each other? (Strong signal of shared subfield)
        # 4. Overlap: Do they share a high percentage of mutual neighbors?
        for neighbor_id in candidates:
            # ! Eliminate the cases with more than 50 citations (Neighbor Hubs)
            # Extremely highly-cited papers are generic background noise
            if in_map.get(neighbor_id, 0.0) > 50.0:
                continue

            recip = reciprocity_value(edge_type(graph, node_id, neighbor_id))
            over = overlap_score(local_contexts, node_id, neighbor_id)

            # ! Eliminate low chances of citing each other
            # If nodes share no mutual neighbors and have no reciprocal edge, skip them
            if over < 0.05 and recip == 0.0:
                continue

            # Switched to Linear normalization (removed np.log1p)
            conn = (in_map.get(neighbor_id, 0.0) + out_map.get(neighbor_id, 0.0)) / max(max_degree, 1e-8)
            n_year = year_lookup.get(neighbor_id)
            temp = (1.0 if node_year is None or n_year is None or
                    np.isnan(node_year) or np.isnan(n_year) else 
                    float(np.exp(-abs(int(node_year) - int(n_year)) / 5.0)))

            node_scores[neighbor_id] = float(
                (connectivity_weight * conn) + (temporal_weight * temp) + 
                (reciprocity_weight * recip) + (overlap_weight * over))
            
        relevance[node_id] = node_scores
    return relevance


def build_neighbor_cache(
    graph: GraphData, node_ids: Iterable[int], documents: pl.DataFrame, max_context_size: int,
    valid_node_ids: Iterable[int] | None = None, sampling_strategy: str = "local_relevance",
    connectivity_weight: float = 0.35, temporal_weight: float = 0.35,
    reciprocity_weight: float = 0.15, overlap_weight: float = 0.15,
    k_hops: int = 2, spectral_dim: int = 0, enable_spectral: bool = False,
) -> NeighborCache:
    """Build the reusable citation-context cache used by the dataset."""
    
    # CPU Pre-computation vs. GPU On-the-fly
    # Graph sampling, spectral decomposition, and BFS hop traversals are CPU-bound 
    # operations. Doing this on-the-fly inside a PyTorch DataLoader would violently 
    # bottleneck the GPU. By fully materializing the context cache beforehand, the 
    # DataLoader only has to perform fast O(1) dictionary lookups during training.
    valid_ids = set(map(int, valid_node_ids)) if valid_node_ids is not None else set(map(int, node_ids))
    node_list = list(map(int, node_ids))
    strat = sampling_strategy.lower()

    if strat == "local_relevance":
        rel_scores = local_relevance_func(graph, node_list, documents, connectivity_weight, temporal_weight, reciprocity_weight, overlap_weight)
    elif strat == "top_k":
        in_deg = degree(graph.edge_index[1], num_nodes=graph.num_nodes).cpu().tolist()
        out_deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes).cpu().tolist()
        in_map = {int(graph.node_ids[i]): float(v) for i, v in enumerate(in_deg)}
        out_map = {int(graph.node_ids[i]): float(v) for i, v in enumerate(out_deg)}
        
        rel_scores = {n: {
            nbr: float(in_map.get(nbr, 0.0) + out_map.get(nbr, 0.0)) 
            for nbr in (
                set(graph.out_neighbors.get(n, set())) |
                set(graph.in_neighbors.get(n, set()))) - {n}
        } for n in node_list}
    else:
        raise ValueError(f"Unknown sampling strategy: {strat}")

    y_lookup = build_year_lookup(documents)
    spec_lookup = spectral_features(graph, node_list, spectral_dim=spectral_dim, enabled=enable_spectral)
    hop_lookup = {n: k_hop_profile(graph, n, max_hops=k_hops) for n in node_list}

    cache: NeighborCache = {}
    
    # Bounded Subgraphs for Transformers
    # Standard GNNs suffer from "neighbor explosion" as layers get deeper. By bounding 
    # the subgraph to `max_context_size` (Top-K selection), guarantee a fixed memory 
    # footprint. Furthermore, because Transformers don't inherently understand graph 
    # topology, explicitly embed structural traits (edge_type, year_delta, spectral, 
    # hop_profile) directly into the cache to act as "Structural Positional Encodings".
    for node_id in node_list:
        ranked = sorted([(nbr, sc) for nbr, sc in rel_scores.get(node_id, {}).items() if nbr in valid_ids and nbr != node_id], key=lambda x: x[1], reverse=True)
        n_year = y_lookup.get(node_id)
        
        cache[node_id] = [{
            "doc_id": int(nbr), 
            "edge_type": int(edge_type(graph, node_id, nbr)),
            "year_delta": 0.0 if n_year is None or y_lookup.get(nbr) is None or np.isnan(n_year) or np.isnan(y_lookup.get(nbr)) else float(y_lookup.get(nbr) - n_year),
            "score": float(sc), 
            "hop_profile": hop_lookup.get(nbr, [0.0] * k_hops), 
            "spectral": spec_lookup.get(nbr, [0.0] * spectral_dim)}
        for nbr, sc in ranked[:max_context_size]]
    return cache


def save_neighbor_cache(cache: NeighborCache, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    """Save a citation-context cache to disk."""
    # JSON Serialization
    # Graph objects (PyG Data) are opaque binary blobs when saved via torch.save().
    # Serializing the sampled subgraphs to human-readable JSON allows researchers to 
    # manually inspect the neighbor selection quality without writing extraction scripts.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata or {}, "cache": {str(k): [dict(e) for e in v] for k, v in cache.items()}}
    path.write_text(json.dumps(payload, indent=2))


def load_neighbor_cache(path: str | Path) -> tuple[NeighborCache, dict[str, Any]]:
    """Load a previously saved citation-context cache."""
    # Research codebases evolve rapidly. If the cache format changes
    # (e.g., adding spectral features later), this loading function detects 
    # old formats and fills missing features with zeros so that old, 
    # expensive preprocessing runs don't have to be thrown away or re-run.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")

    payload = json.loads(path.read_text())
    cache: NeighborCache = {}

    for node_id, node_data in payload.get("cache", {}).items():
        if isinstance(node_data, list):
            cache[int(node_id)] = [{
                "doc_id": int(e.get("doc_id", 0)),
                "edge_type": int(e.get("edge_type", int(EdgeType.NONE))),
                "year_delta": float(e.get("year_delta", 0.0)), 
                "score": float(e.get("score", 0.0)),
                "hop_profile": [float(v) for v in e.get("hop_profile", [])], 
                "spectral": [float(v) for v in e.get("spectral", [])]}
            for e in node_data]
        else:
            seen = set()
            cache[int(node_id)] = []
            for group in node_data.values():
                for nbr in map(int, group):
                    if nbr not in seen:
                        seen.add(nbr)
                        cache[int(node_id)].append({
                            "doc_id": nbr,
                            "edge_type": int(EdgeType.NONE), 
                            "year_delta": 0.0, "score": 0.0, 
                            "hop_profile": [], "spectral": []})

    return cache, payload.get("metadata", {})
