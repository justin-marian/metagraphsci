"""
Neighbour-cache construction utilities for citation-context modelling.

The cache moves expensive graph work traversal, structural scoring, optional
spectral feature computation, and BFS hop profiling out of the training loop
so that each DataLoader read is a cheap dictionary lookup and slice operation.

NeighborCache: a dict mapping each center document id to a ranked list of neighbour records.  
Persistence with backward-compatible format detection, and flexible scoring strategies with
tunable structural and temporal weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl
from torch_geometric.utils import degree

from constants import (
    CONNECTIVITY_WEIGHT,
    OVERLAP_WEIGHT,
    RECIPROCITY_WEIGHT,
    TEMPORAL_WEIGHT,
    HUB_DEGREE_THR,
    K_HOPS,
    SPECTRAL_DIM,
    MAX_GRAPH_NODES_FOR_HOPS,
    TEMPORAL_DECAY_YEARS,
    EdgeType, GraphData, NeighborCache)
from graph_utils import (
    build_local_context_map, edge_type,
    k_hop_profile, overlap_score,
    reciprocity_value, spectral_features)
from tabular_utils import build_year_lookup


def missing_year(value: Any) -> bool:
    """
    Return True when a value year should be treated as unavailable.

    The pipeline receives years from CSV and JSON sources where values may be
    empty strings, Python None, or float NaN.  Normalising all those cases
    here keeps the temporal scoring logic simple and purely numeric downstream.
    """
    if value is None:
        return True
    try:
        # Cast to float first so integer strings like "2005" work correctly,
        # while truly missing values (empty string, "nan") raise or return NaN.
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return True


def year_delta(node_year: Any, neighbor_year: Any) -> float:
    """
    Return the signed temporal distance: neighbor_year − node_year.

    Positive means the neighbour was published later than the center node
    (a future citation); negative means the neighbour is older (a reference).

    Returns 0.0 when either year is unavailable so the downstream tensor
    receives a neutral value rather than NaN.
    """
    if missing_year(node_year) or missing_year(neighbor_year):
        return 0.0
    return float(int(neighbor_year) - int(node_year))


def time_similarity(node_year: Any, neighbor_year: Any) -> float:
    """
    Convert year distance into a smooth exponential similarity score.

    Uses an exponential decay with half-life TEMPORAL_DECAY_YEARS.  
    Papers published in the same year score 1.0; papers separated by 
    TEMPORAL_DECAY_YEARS score => exp(-1) ~~ 0.37. 
    Missing years yield 1.0 (neutral) so they do not unfairly penalise nodes with sparse metadata.
    """
    if missing_year(node_year) or missing_year(neighbor_year):
        # Neutral score: we cannot penalise a pair for missing metadata.
        return 1.0
    return float(np.exp(-abs(int(node_year) - int(neighbor_year)) / TEMPORAL_DECAY_YEARS))


def node_degree_maps(graph: GraphData) -> tuple[dict[int, float], dict[int, float], float]:
    """
    Build node-id-keyed in/out degree maps and a shared normalisation scale.

    PyG stores edges in dense tensor index space while cache entries are keyed
    by external document ids, so the mapping from tensor index → doc_id is
    centralised here rather than duplicated across scoring functions.

    in_map       –  {doc_id: in_degree}
    out_map      –  {doc_id: out_degree}
    max_degree   –  max(all in/out degrees), 
                    used to normalise connectivity scores into [0, 1].
                    At least 1.0 to guard against div-by-zero on disconnected singleton graphs.
    """
    in_degree  = degree(graph.edge_index[1], num_nodes=graph.num_nodes).cpu().tolist()
    out_degree = degree(graph.edge_index[0], num_nodes=graph.num_nodes).cpu().tolist()

    in_map  = {int(graph.node_ids[i]): float(v) for i, v in enumerate(in_degree)}
    out_map = {int(graph.node_ids[i]): float(v) for i, v in enumerate(out_degree)}

    # Hard floor so we never divide by zero: 1.0 - neutral connectivity score for isolated nodes.
    max_degree = float(max(max(in_map.values(),  default=1.0), max(out_map.values(), default=1.0), 1.0))
    return in_map, out_map, max_degree


def direct_neighbors(graph: GraphData, node_id: int) -> set[int]:
    """
    Return the union of incoming and outgoing neighbours for one node.

    Self-loops are excluded because scoring a node against itself is
    meaningless and pollutes the cache with useless entries.
    """
    return graph.out_neighbors.get(node_id, set()) | graph.in_neighbors.get(node_id, set()) - {node_id}


def top_k_scores(graph: GraphData, node_ids: list[int]) -> dict[int, dict[int, float]]:
    """
    Score neighbours using raw total degree only.

    This is the lightweight baseline strategy: higher combined in+out degree
    ranks higher, with no temporal, reciprocity, or overlap terms.  Use it
    when scoring speed matters more than neighbourhood quality.
    """
    in_map, out_map, _ = node_degree_maps(graph)
    return {
        node_id: {neighbor_id: float(in_map[neighbor_id] + out_map[neighbor_id])
        for neighbor_id in direct_neighbors(graph, node_id)
    } for node_id in node_ids}


def local_relevance_func(
    graph: GraphData, node_ids: Iterable[int], documents: pl.DataFrame,
    # Tune these to shift the relative importance of different signals in the final cache ranking.
    connectivity_weight: float, temporal_weight: float, reciprocity_weight: float, overlap_weight: float,
    hub_degree_THR: int = HUB_DEGREE_THR) -> dict[int, dict[int, float]]:
    """
    Score candidate citations with a compact structural–temporal relevance model.
    The score for each (center, neighbor) pair is a weighted sum of four terms:

    Tunning weights
    ----------------
    connectivity  – normalised total degree of the neighbour; rewards well-cited papers that are likely to be important.
    temporal      – exponential decay of year distance; rewards papers published close in time to the center node.
    reciprocity   – 1.0 for bidirectional citations, 0.0 otherwise; rewards mutual references which indicate intellectual kinship.
    overlap       – Jaccard overlap of one-hop neighbourhoods; rewards papers that share many co-citations, indicating topical proximity.

    Hub filtering
    ----------------
    Survey papers and textbooks often dominate local neighbourhoods with generic, low-value citations. 

    When degree of the hub > defined threshold, any node center or neighbour whose in-degree exceeds
    the threshold is excluded from scoring entirely.
    """
    year_lookup    = build_year_lookup(documents)
    local_contexts = build_local_context_map(graph)
    in_map, out_map, max_degree = node_degree_maps(graph)
    relevance: dict[int, dict[int, float]] = {}

    for node_id in map(int, node_ids):
        # Skip hub nodes entirely when the threshold is active.
        if hub_degree_THR > 0 and in_map[node_id] > hub_degree_THR:
            continue

        candidates = local_contexts[node_id]
        if not candidates:
            continue

        node_year = year_lookup[node_id]
        scores: dict[int, float] = {}

        for neighbor_id in candidates:
            # Skip hub neighbours for the same reason as hub center nodes.
            if hub_degree_THR > 0 and in_map[neighbor_id] > hub_degree_THR:
                continue

            # Connectivity is the sum of in- and out-degree normalised by the
            # maximum degree in the graph so scores are comparable across graphs with different edge densities.
            connectivity = in_map[neighbor_id] + out_map[neighbor_id] / max_degree
            scores[neighbor_id] = (
                connectivity_weight  * connectivity
                + temporal_weight    * time_similarity(node_year, year_lookup[neighbor_id])
                + reciprocity_weight * reciprocity_value(edge_type(graph, node_id, neighbor_id))
                + overlap_weight     * overlap_score(local_contexts, node_id, neighbor_id))

        if scores:
            relevance[node_id] = scores

    return relevance


def build_relevance_scores(
    graph: GraphData,
    node_ids: list[int],
    documents: pl.DataFrame,
    sampling_strategy: str,
    # Weights for the local relevance strategy; ignored by other strategies.
    connectivity_weight: float,
    temporal_weight: float,
    reciprocity_weight: float,
    overlap_weight: float,
    # Structural feature options (ignored when k hops are 0 or spectral dim is 0 or not enabled). 
    # These do not affect the relevance scores themselves but are included here to avoid redundant 
    # graph traversals when both scores and features are needed for cache entries.
    hub_degree_THR: int
) -> dict[int, dict[int, float]]:
    """Dispatch to the configured neighbour-scoring strategy.

    Supported strategies
    --------------------
    "local_relevance"  – Weighted structural+temporal model (recommended).
    "top_k"            – Raw degree ranking (fast baseline, no temporal signal).

    Raises ValueError for unrecognised strategy strings.
    """
    strategy = sampling_strategy.lower()
    if strategy == "local_relevance":
        return local_relevance_func(
            graph=graph, node_ids=node_ids, documents=documents,
            # Tuning weights for the local relevance strategy; ignored by other strategies.
            connectivity_weight=connectivity_weight,
            temporal_weight=temporal_weight,
            reciprocity_weight=reciprocity_weight,
            overlap_weight=overlap_weight,
            # Structural feature options (ignored when k hops are 0 or spectral dim is 0 or not enabled).
            hub_degree_THR=hub_degree_THR)

    if strategy == "top_k":
        return top_k_scores(graph, node_ids)

    raise ValueError(f"Unknown sampling strategy: {strategy!r}")


def neighbor_entry(
    graph: GraphData, node_id: int, neighbor_id: int, score: float,
    year_lookup: dict[int, Any], hop_lookup: dict[int, list[float]], spectral_lookup: dict[int, list[float]]
) -> dict[str, Any]:
    """
    Build the serialised cache record for one (center, neighbour) pair.

    The record always contains: doc_id, edge_type, year_delta, score.

    Optional fields like the profile of the hops and spectral are added only when the
    corresponding lookup tables contain an entry for neighbor_id, so callers must check for their presence with.
    """
    entry: dict[str, Any] = {
        "doc_id":     int(neighbor_id),
        "edge_type":  int(edge_type(graph, node_id, neighbor_id)),
        "year_delta": year_delta(year_lookup.get(node_id), year_lookup.get(neighbor_id)),
        "score":      float(score)
    }
    # Append optional structural features only when they were computed.
    if hop_profile := hop_lookup.get(neighbor_id):
        entry["hop_profile"] = hop_profile
    if spectral := spectral_lookup.get(neighbor_id):
        entry["spectral"] = spectral
    return entry


def build_neighbor_cache(
    graph: GraphData, node_ids: Iterable[int], documents: pl.DataFrame,
    max_context_size: int, valid_node_ids: Iterable[int] | None = None,
    sampling_strategy: str = "local_relevance",
    # Weights for the local relevance strategy; ignored by other strategies.
    # Tune these to shift the relative importance of different signals in the final cache ranking.
    connectivity_weight: float = CONNECTIVITY_WEIGHT,
    temporal_weight:     float = TEMPORAL_WEIGHT,
    reciprocity_weight:  float = RECIPROCITY_WEIGHT,
    overlap_weight:      float = OVERLAP_WEIGHT,
    # Structural feature options (ignored when k_hops=0 or spectral_dim=0 or enable_spectral=False).
    enable_spectral: bool = False,
    k_hops: int = K_HOPS,
    spectral_dim: int = SPECTRAL_DIM,
    hub_degree_THR: int = HUB_DEGREE_THR,
    # Safety cap for BFS traversal
    max_graph_nodes_for_hops: int = MAX_GRAPH_NODES_FOR_HOPS
) -> NeighborCache:
    """
    Materialise the reusable ego-context cache consumed during training.

    All expensive graph operations happen once here on CPU during preprocessing. 
    The resulting JSON-friendly dict keeps the DataLoader fast and reproducible across training runs.

    Parameters
    ----------
    graph                       : Citation graph with node ids mapped to document ids in the same space as the documents.
    node_ids                    : Center nodes for which to build cache entries, typically the training split node ids.
    documents                   : Full document metadata (used for year lookups), node ids must be present.
    max_context_size            : Maximum neighbours to store per center node, ranked by relevance score.
                                  Entries are ranked by score; lower-ranked ones are discarded.
    valid_node_ids              : Superset of node ids allowed as neighbours.
                                  Use this to restrict cache entries to the training split in inductive settings.
    sampling_strategy           : "local_relevance" (default) or "top_k" (fast degree-based baseline).
    connectivity_weight         : Weight for degree-based connectivity term.
    temporal_weight             : Weight for exponential temporal similarity term.
    reciprocity_weight          : Weight for bidirectional-citation term.
    overlap_weight              : Weight for Jaccard neighbourhood overlap term.
    max_graph_nodes_for_hops    : BFS is skipped when graph exceeds this size.

    Hop-profile and spectral features are optional because they require additional graph traversals and matrix computations.  
    When disabled, cache entries simply omit those fields so the dataset layer can handle them with default values.

    k_hops                      : BFS depth for hop-profile feature (0 = disabled, no BFS, no hop features).
    spectral_dim                : Laplacian eigenvector dimension (0 = disabled, no spectral features).
    enable_spectral             : Must be True AND spectral_dim > 0 to compute and include spectral features.
    hub_degree_THR        : Exclude nodes with in-degree > this (0 = off, no hub filtering).
    """
    node_list = [int(node_id) for node_id in node_ids]
    # Restrict valid neighbours to the provided set; default to all center nodes.
    valid_ids = {int(nid) for nid in valid_node_ids} if valid_node_ids is not None else set(node_list)
    relevance_scores = build_relevance_scores(
        graph=graph, node_ids=node_list, documents=documents,
        sampling_strategy=sampling_strategy,
        # Weights for the local relevance strategy; ignored by other strategies.
        connectivity_weight=connectivity_weight,
        temporal_weight=temporal_weight,
        reciprocity_weight=reciprocity_weight,
        overlap_weight=overlap_weight,
        # Structural feature options (ignored when k hops are 0 or spectral dim is 0 or not enabled).
        hub_degree_THR=hub_degree_THR)

    year_lookup     = build_year_lookup(documents)
    spectral_lookup = spectral_features(graph, node_list, spectral_dim=spectral_dim, enabled=enable_spectral)

    # BFS hop profiles are only computed when the graph is small enough to make
    # the traversal tractable. Oversized graphs simply produce no hop entries.
    hop_lookup: dict[int, list[float]] = (
        {node_id: k_hop_profile(graph, node_id, max_hops=k_hops) for node_id in node_list}
        if k_hops > 0 and len(node_list) <= max_graph_nodes_for_hops else {})

    cache: NeighborCache = {}
    for node_id in node_list:
        # Filter to valid neighbours; exclude the center node from its own list.
        scored = {
            neighbor_id: score
            for neighbor_id, score in relevance_scores[node_id].items()
            if neighbor_id in valid_ids and neighbor_id != node_id
        }

        if not scored:
            # No valid scored neighbours omit this node rather than producing
            # a cache entry with fabricated zero-signal content.
            continue

        # Sort descending by score and keep only the top max_context_size entries.
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        cache[node_id] = [neighbor_entry(
            graph=graph, node_id=node_id, neighbor_id=neighbor_id, score=score, 
            year_lookup=year_lookup, hop_lookup=hop_lookup, spectral_lookup=spectral_lookup)
        for neighbor_id, score in ranked[:max_context_size]]

    return cache


def save_neighbor_cache(cache: NeighborCache, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    """
    Persist the neighbour cache as human-readable JSON.

    JSON is deliberately chosen over a binary format because cache quality
    often needs manual inspection during dataset debugging and ablation work.

    The metadata dict can carry build parameters (weights, strategy, etc.) for experiment reproducibility.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata or {},
        # Stringify integer keys because JSON object keys must be strings.
        "cache": {
            str(node_id): [dict(entry) for entry in entries]
            for node_id, entries in cache.items()
        }
    }
    output_path.write_text(json.dumps(payload, indent=2))


def load_list_cache_entries(node_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalise the modern flat-list cache format to strict numeric fields.

    Enforces types (int doc_id, int edge_type, float year_delta, float score)
    so the dataset layer never has to deal with strings or None from the JSON deserialiser.
    """
    entries = []
    for entry in node_data:
        parsed: dict[str, Any] = {
            "doc_id":     int(entry["doc_id"]),
            "edge_type":  int(entry["edge_type"]),
            "year_delta": float(entry["year_delta"]),
            "score":      float(entry["score"])
        }
        if hp := entry.get("hop_profile"):
            parsed["hop_profile"] = [float(v) for v in hp]
        if sp := entry.get("spectral"):
            parsed["spectral"] = [float(v) for v in sp]
        entries.append(parsed)
    return entries


def load_legacy_cache_entries(node_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Upgrade the old dict-of-hop-groups layout to the modern flat-list format.

    The legacy format stored {hop_key: [neighbor_id, ...]} dicts.

    It lacked edge types, year deltas, and scores, so the loader fills neutral defaults
    (NONE edge type, 0.0 for numeric fields) rather than fabricating values that would look meaningful.
    """
    seen: set[int] = set()
    entries: list[dict[str, Any]] = []

    for group in node_data.values():
        for neighbor_id in map(int, group):
            if neighbor_id in seen:
                continue  # Deduplicate across hop groups.
            seen.add(neighbor_id)
            entries.append({"doc_id": neighbor_id, "edge_type": int(EdgeType.NONE), "year_delta": 0.0, "score": 0.0})

    return entries


def load_neighbor_cache(path: str | Path) -> tuple[NeighborCache, dict[str, Any]]:
    """
    Load a saved neighbour cache with backward-compatible format detection.

    Supports both the modern flat-list format and the legacy dict-of-hop-groups
    format written by older pipeline versions.

    Returns the cache dict and the metadata dict stored alongside it.
    """
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    payload = json.loads(cache_path.read_text())
    cache: NeighborCache = {}

    for node_id, node_data in payload["cache"].items():
        cache[int(node_id)] = (
            # Modern format: each node's value is a list of entry dicts.
            load_list_cache_entries(node_data) if isinstance(node_data, list)
            # Legacy format: each node's value is a dict of hop-group lists.
            else load_legacy_cache_entries(node_data))

    return cache, payload["metadata"]
