"""
Neighbour-cache construction utilities for citation-context modelling.

The cache moves expensive graph work, structural scoring, optional spectral feature
computation, and BFS hop profiling out of the training loop so each DataLoader read
is a cheap dictionary lookup and slice operation.

NeighborCache maps each center document id to a ranked list of neighbour records.
Persistence supports backward-compatible format detection and flexible scoring
strategies with tunable structural and temporal weights.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from loguru import logger
from torch_geometric.utils import degree

from .cache_utils import (
    docs_fingerprint, edge_set_fingerprint, metadata_matches, stable_int_fingerprint)
from .constants import (
    CONNECTIVITY_WEIGHT, HUB_DEGREE_THR, K_HOPS, MAX_GRAPH_NODES_FOR_HOPS,
    OVERLAP_WEIGHT, RECIPROCITY_WEIGHT, SPECTRAL_DIM, TEMPORAL_DECAY_YEARS,
    TEMPORAL_WEIGHT, EdgeType, GraphData, NeighborCache)
from .graph_utils import build_local_context_map, build_undirected_neighbors, edge_type, spectral_features
from .tabular_utils import build_year_lookup


def resolve_n_jobs(n_jobs: int) -> int:
    """Translate joblib-style n_jobs into a concrete worker count."""
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return n_jobs


def chunk_indices(n_items: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split ``[0, n_items)`` into contiguous ``(start, stop)`` ranges."""
    if n_items == 0 or n_chunks <= 1:
        return [(0, n_items)]

    n_chunks = min(n_chunks, n_items)
    base, extra = divmod(n_items, n_chunks)
    ranges: list[tuple[int, int]] = []
    start = 0
    for idx in range(n_chunks):
        stop = start + base + (1 if idx < extra else 0)
        ranges.append((start, stop))
        start = stop
    return ranges


def missing_year(value: Any) -> bool:
    """
    Return True when a publication year should be treated as unavailable.

    CSV and JSON sources may encode missing years as empty strings, None, or NaN.
    Normalising those cases here keeps temporal scoring numeric downstream.
    """
    if value is None:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return True


def year_delta(node_year: Any, neighbor_year: Any) -> float:
    """
    Return signed temporal distance ``neighbor_year - node_year``.

    Positive means the neighbour was published later than the center node.
    Missing years produce 0.0 so downstream tensors never receive NaN.
    """
    if missing_year(node_year) or missing_year(neighbor_year):
        return 0.0
    return float(int(neighbor_year) - int(node_year))


def time_similarity(node_year: Any, neighbor_year: Any, decay: float = TEMPORAL_DECAY_YEARS) -> float:
    """
    Convert year distance into smooth exponential similarity.

    Same-year papers score 1.0. Papers separated by TEMPORAL_DECAY_YEARS score
    approximately exp(-1). Missing years yield neutral score 1.0.
    """
    if missing_year(node_year) or missing_year(neighbor_year):
        return 1.0
    return float(np.exp(-abs(int(node_year) - int(neighbor_year)) / float(decay)))


def node_degree_maps(graph: GraphData) -> tuple[dict[int, float], dict[int, float], float]:
    """
    Build node-id-keyed in/out degree maps and a shared normalisation scale.

    PyG stores edges in dense tensor index space while cache entries are keyed by
    external document ids, so tensor index -> doc_id conversion is centralised here.
    """
    in_degree = degree(graph.edge_index[1], num_nodes=graph.num_nodes).cpu().tolist()
    out_degree = degree(graph.edge_index[0], num_nodes=graph.num_nodes).cpu().tolist()

    in_map = {int(graph.node_ids[idx]): float(value) for idx, value in enumerate(in_degree)}
    out_map = {int(graph.node_ids[idx]): float(value) for idx, value in enumerate(out_degree)}

    # Hard floor prevents div-by-zero on disconnected singleton graphs.
    max_degree = float(max(max(in_map.values(), default=1.0), max(out_map.values(), default=1.0), 1.0))
    return in_map, out_map, max_degree


def direct_neighbors(graph: GraphData, node_id: int) -> set[int]:
    """Return incoming ∪ outgoing neighbours for one node, excluding self-loops."""
    return (graph.out_neighbors.get(node_id, set()) | graph.in_neighbors.get(node_id, set())) - {node_id}


def top_k_scores(graph: GraphData, node_ids: list[int]) -> dict[int, dict[int, float]]:
    """
    Score neighbours using raw total degree only.

    This lightweight baseline ranks by combined in+out degree with no temporal,
    reciprocity, or overlap terms.
    """
    in_map, out_map, _ = node_degree_maps(graph)
    return {node_id: {
        neighbor_id: float(in_map[neighbor_id] + out_map[neighbor_id])
        for neighbor_id in direct_neighbors(graph, node_id)}
        for node_id in node_ids}


def score_chunk(
    node_chunk: list[int],
    in_map: dict[int, float], out_map: dict[int, float], max_degree: float,
    local_contexts: dict[int, set[int]], year_lookup: dict[int, Any],
    edge_set: set[tuple[int, int]], weights: tuple[float, float, float, float],
    hub_thr: int) -> list[tuple[int, dict[int, float]]]:
    """
    Score a slice of center nodes.

    Kept top-level and picklable for loky workers. The expression order mirrors
    the serial path so serial and parallel output remain deterministic.
    """
    connectivity_weight, temporal_weight, reciprocity_weight, overlap_weight = weights
    results: list[tuple[int, dict[int, float]]] = []

    for node_id in node_chunk:
        if hub_thr > 0 and in_map[node_id] > hub_thr:
            continue

        candidates = local_contexts[node_id]
        if not candidates:
            continue

        node_year = year_lookup[node_id]
        node_ctx = local_contexts.get(node_id, set())
        scores: dict[int, float] = {}

        for neighbor_id in candidates:
            if hub_thr > 0 and in_map[neighbor_id] > hub_thr:
                continue

            neighbor_ctx = local_contexts.get(neighbor_id, set())
            union = node_ctx | neighbor_ctx
            connectivity = (in_map[neighbor_id] + out_map[neighbor_id]) / max_degree
            reciprocity = 1.0 if edge_type(edge_set, node_id, neighbor_id) == EdgeType.BIDIRECTIONAL else 0.0
            overlap = len(node_ctx & neighbor_ctx) / len(union) if union else 0.0
            temporal = time_similarity(node_year, year_lookup.get(neighbor_id, node_year))

            scores[neighbor_id] = (
                connectivity_weight * connectivity + temporal_weight * temporal +
                reciprocity_weight * reciprocity + overlap_weight * overlap)

        if scores:
            results.append((node_id, scores))

    return results


def local_relevance_func(
    graph: GraphData, node_ids: Iterable[int], documents: pl.DataFrame,
    connectivity_weight: float, temporal_weight: float,
    reciprocity_weight: float, overlap_weight: float,
    hub_degree_threshold: int = HUB_DEGREE_THR,
    *, n_jobs: int = -1) -> dict[int, dict[int, float]]:
    """
    Score candidate citations with a compact structural-temporal relevance model.

    The score for each ``(center, neighbor)`` pair is a weighted sum of:
    connectivity, temporal similarity, reciprocity, and neighbourhood overlap.

    Hub filtering excludes center or neighbour nodes whose in-degree exceeds the
    configured threshold. A threshold of 0 disables hub filtering.
    """
    year_lookup = build_year_lookup(documents)
    local_contexts = build_local_context_map(graph)
    in_map, out_map, max_degree = node_degree_maps(graph)
    weights = (connectivity_weight, temporal_weight, reciprocity_weight, overlap_weight)

    node_list = [int(node_id) for node_id in node_ids]
    workers = resolve_n_jobs(n_jobs)

    if workers == 1 or len(node_list) <= 1:
        chunk_results = [score_chunk(
            node_list, in_map, out_map, max_degree, local_contexts,
            year_lookup, graph.edge_set, weights, hub_degree_threshold)]
    else:
        logger.info("Building neighbor cache scores with n_jobs={} over {} nodes (parallel)", workers, len(node_list))
        ranges = chunk_indices(len(node_list), workers)
        chunk_results = Parallel(n_jobs=workers, backend="loky", verbose=0)(
            delayed(score_chunk)(
                node_list[start:stop], in_map, out_map, max_degree, local_contexts,
                year_lookup, graph.edge_set, weights, hub_degree_threshold)
            for start, stop in ranges)

    # Parallel preserves submission order; contiguous chunks preserve node ordering.
    relevance: dict[int, dict[int, float]] = {}
    for chunk in chunk_results:
        for node_id, scores in chunk:
            relevance[node_id] = scores

    return relevance


def build_relevance_scores(
    graph: GraphData, node_ids: list[int], documents: pl.DataFrame,
    sampling_strategy: str, connectivity_weight: float,
    temporal_weight: float, reciprocity_weight: float,
    overlap_weight: float, hub_degree_threshold: int,
    *, n_jobs: int = -1) -> dict[int, dict[int, float]]:
    """
    Dispatch to the configured neighbour-scoring strategy.

    Supported strategies:
    - ``local_relevance``: weighted structural + temporal model.
    - ``top_k``: raw degree ranking baseline.
    """
    strategy = sampling_strategy.lower()

    if strategy == "local_relevance":
        return local_relevance_func(
            graph, node_ids, documents,
            connectivity_weight=connectivity_weight, temporal_weight=temporal_weight,
            reciprocity_weight=reciprocity_weight, overlap_weight=overlap_weight,
            hub_degree_threshold=hub_degree_threshold, n_jobs=n_jobs)

    if strategy == "top_k":
        return top_k_scores(graph, node_ids)

    raise ValueError(f"Unknown sampling strategy: {strategy!r}")


def hop_chunk(node_chunk: list[int], undirected: dict[int, set[int]], max_hops: int) -> list[tuple[int, list[float]]]:
    """BFS hop profiles for a slice of nodes using a precomputed undirected map."""
    results: list[tuple[int, list[float]]] = []

    for node_id in node_chunk:
        if max_hops <= 0 or node_id not in undirected:
            results.append((node_id, []))
            continue

        visited = {node_id}
        frontier = {node_id}
        counts = [0.0] * max_hops

        for hop in range(max_hops):
            next_frontier = set().union(*(undirected.get(node, set()) for node in frontier)) - visited
            counts[hop] = float(len(next_frontier))

            if not next_frontier:
                break

            visited |= next_frontier
            frontier = next_frontier

        total = sum(counts)
        results.append((node_id, [count / total for count in counts] if total > 0 else counts))

    return results


def assemble_chunk(
    node_chunk: list[int], relevance_scores: dict[int, dict[int, float]],
    edge_set: set[tuple[int, int]], year_lookup: dict[int, Any],
    hop_lookup: dict[int, list[float]], spectral_lookup: dict[int, list[float]],
    valid_ids: set[int], max_context_size: int) -> list[tuple[int, list[dict[str, Any]]]]:
    """Build serialisable cache entries for a slice of center nodes."""
    results: list[tuple[int, list[dict[str, Any]]]] = []

    for node_id in node_chunk:
        scored = {
            neighbor_id: score
            for neighbor_id, score in relevance_scores.get(node_id, {}).items()
            if neighbor_id in valid_ids and neighbor_id != node_id}

        if not scored:
            continue

        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        node_year = year_lookup.get(node_id)
        entries: list[dict[str, Any]] = []

        for neighbor_id, score in ranked[:max_context_size]:
            neighbor_year = year_lookup.get(neighbor_id)
            entry: dict[str, Any] = {
                "doc_id": int(neighbor_id),
                "edge_type": int(edge_type(edge_set, node_id, neighbor_id)),
                "year_delta": year_delta(node_year, neighbor_year),
                "score": float(score)}

            if hop_profile := hop_lookup.get(neighbor_id):
                entry["hop_profile"] = hop_profile
            if spectral := spectral_lookup.get(neighbor_id):
                entry["spectral"] = spectral

            entries.append(entry)

        results.append((node_id, entries))

    return results


def build_neighbor_cache(
    graph: GraphData, node_ids: Iterable[int], documents: pl.DataFrame,
    max_context_size: int, valid_node_ids: Iterable[int] | None = None,
    sampling_strategy: str = "local_relevance",
    connectivity_weight: float = CONNECTIVITY_WEIGHT,
    temporal_weight: float = TEMPORAL_WEIGHT,
    reciprocity_weight: float = RECIPROCITY_WEIGHT,
    overlap_weight: float = OVERLAP_WEIGHT,
    enable_spectral: bool = False, k_hops: int = K_HOPS,
    spectral_dim: int = SPECTRAL_DIM,
    hub_degree_threshold: int = HUB_DEGREE_THR,
    max_graph_nodes_for_hops: int = MAX_GRAPH_NODES_FOR_HOPS,
    n_jobs: int = -1) -> NeighborCache:
    """
    Materialise the reusable ego-context cache consumed during training.

    Expensive graph operations happen once during preprocessing. The resulting
    JSON-friendly dict keeps DataLoader reads fast and reproducible.

    ``valid_node_ids`` restricts allowed neighbours, typically for inductive
    settings. Hop-profile and spectral features are optional and omitted from
    entries when disabled.
    """
    node_list = [int(node_id) for node_id in node_ids]
    valid_ids = {int(node_id) for node_id in valid_node_ids} if valid_node_ids is not None else set(node_list)
    workers = resolve_n_jobs(n_jobs)

    logger.info("Building neighbor cache with n_jobs={} over {} nodes (parallel)",
                workers, len(node_list))

    relevance_scores = build_relevance_scores(
        graph, node_list, documents, sampling_strategy,
        connectivity_weight=connectivity_weight, temporal_weight=temporal_weight,
        reciprocity_weight=reciprocity_weight, overlap_weight=overlap_weight,
        hub_degree_threshold=hub_degree_threshold, n_jobs=n_jobs)

    year_lookup = build_year_lookup(documents)
    spectral_lookup = spectral_features(graph, node_list, spectral_dim=spectral_dim, enabled=enable_spectral)

    # BFS hop profiles are skipped on large graphs to avoid blocking cache construction.
    hop_lookup: dict[int, list[float]] = {}
    if k_hops > 0 and len(node_list) <= max_graph_nodes_for_hops:
        undirected = build_undirected_neighbors(graph)

        if workers == 1 or len(node_list) <= 1:
            hop_results = [hop_chunk(node_list, undirected, k_hops)]
        else:
            logger.info("Computing {}-hop profiles with n_jobs={} over {} nodes", k_hops, workers, len(node_list))
            ranges = chunk_indices(len(node_list), workers)
            hop_results = Parallel(n_jobs=workers, backend="loky", verbose=0)(
                delayed(hop_chunk)(node_list[start:stop], undirected, k_hops)
                for start, stop in ranges)

        for chunk in hop_results:
            for node_id, profile in chunk:
                hop_lookup[node_id] = profile

    if workers == 1 or len(node_list) <= 1:
        assembly_results = [assemble_chunk(
            node_list, relevance_scores, graph.edge_set, year_lookup, hop_lookup,
            spectral_lookup, valid_ids, max_context_size)]
    else:
        logger.info("Assembling cache entries with n_jobs={} over {} nodes", workers, len(node_list))
        ranges = chunk_indices(len(node_list), workers)
        assembly_results = Parallel(n_jobs=workers, backend="loky", verbose=0)(
            delayed(assemble_chunk)(
                node_list[start:stop], relevance_scores, graph.edge_set, year_lookup,
                hop_lookup, spectral_lookup, valid_ids, max_context_size)
            for start, stop in ranges)

    cache: NeighborCache = {}
    for chunk in assembly_results:
        for node_id, entries in chunk:
            cache[node_id] = entries

    # Key-sorted JSON makes the digest invariant to insertion order.
    digest_payload = json.dumps({str(key): cache[key] for key in sorted(cache)}, sort_keys=True, separators=(",", ":"))
    logger.info("Neighbor cache built: {} entries, sha256={}", len(cache), hashlib.sha256(digest_payload.encode()).hexdigest())
    return cache


def save_neighbor_cache(
    cache: NeighborCache, path: str | Path,
    metadata: dict[str, Any] | None = None) -> None:
    """
    Persist the neighbour cache as human-readable JSON.

    JSON is used deliberately because cache quality often needs manual inspection
    during dataset debugging and ablation work.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata or {},
        "cache": {str(node_id): [dict(entry) for entry in entries] for node_id, entries in cache.items()}}

    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved neighbor cache: {} centers -> {}", len(cache), output_path)


def load_list_cache_entries(node_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise the modern flat-list cache format to strict numeric fields."""
    entries: list[dict[str, Any]] = []

    for entry in node_data:
        parsed: dict[str, Any] = {
            "doc_id": int(entry["doc_id"]), "edge_type": int(entry["edge_type"]),
            "year_delta": float(entry["year_delta"]), "score": float(entry["score"])}

        if hop_profile := entry.get("hop_profile"):
            parsed["hop_profile"] = [float(value) for value in hop_profile]
        if spectral := entry.get("spectral"):
            parsed["spectral"] = [float(value) for value in spectral]

        entries.append(parsed)

    return entries


def load_legacy_cache_entries(node_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Upgrade the old dict-of-hop-groups layout to the modern flat-list format.

    The legacy format lacks edge types, year deltas, and scores, so neutral
    defaults are used rather than fabricating meaningful-looking values.
    """
    seen: set[int] = set()
    entries: list[dict[str, Any]] = []

    for group in node_data.values():
        for neighbor_id in map(int, group):
            if neighbor_id in seen:
                continue

            seen.add(neighbor_id)
            entries.append({
                "doc_id": neighbor_id, "edge_type": int(EdgeType.NONE),
                "year_delta": 0.0, "score": 0.0})

    return entries


def load_neighbor_cache(path: str | Path, expected_metadata: dict[str, Any] | None = None) -> tuple[NeighborCache, dict[str, Any]]:
    """
    Load a saved neighbour cache with backward-compatible format detection.

    Supports the modern flat-list format and the legacy dict-of-hop-groups format.
    If expected metadata is provided, incomplete or mismatched metadata is rejected.
    """
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    payload = json.loads(cache_path.read_text())
    cache: NeighborCache = {}

    for node_id, node_data in payload["cache"].items():
        cache[int(node_id)] = (
            load_list_cache_entries(node_data) if isinstance(node_data, list)
            else load_legacy_cache_entries(node_data))

    metadata = payload.get("metadata", {})
    if expected_metadata is not None and not neighbor_is_compatible(metadata, expected_metadata):
        raise ValueError(f"Neighbor cache is incompatible with current settings: {cache_path}")

    logger.info("Loaded neighbor cache: {} centers <- {}", len(cache), cache_path)
    return cache, metadata


COMPATIBILITY_KEYS = (
    "num_graph_nodes", "num_center_nodes", "center_node_ids_fingerprint",
    "valid_node_ids_fingerprint", "docs_fingerprint", "citation_edges_fingerprint",
    "max_context_size", "sampling_strategy", "connectivity_weight", "temporal_weight",
    "reciprocity_weight", "overlap_weight", "enable_spectral", "requested_enable_spectral",
    "k_hops", "spectral_dim", "hub_degree_threshold", "max_graph_nodes_for_hops")


def auto_disable_spectral(num_nodes: int, threshold: int = MAX_GRAPH_NODES_FOR_HOPS) -> bool:
    """Return True when spectral features should be skipped for graph-size safety."""
    return int(num_nodes) > int(threshold)


def compute_neighbor_metadata(
    graph: GraphData, documents: pl.DataFrame, node_ids: Iterable[int],
    max_context_size: int, valid_node_ids: Iterable[int] | None = None,
    sampling_strategy: str = "local_relevance",
    connectivity_weight: float = CONNECTIVITY_WEIGHT,
    temporal_weight: float = TEMPORAL_WEIGHT,
    reciprocity_weight: float = RECIPROCITY_WEIGHT,
    overlap_weight: float = OVERLAP_WEIGHT,
    enable_spectral: bool = False, k_hops: int = K_HOPS,
    spectral_dim: int = SPECTRAL_DIM,
    hub_degree_threshold: int = HUB_DEGREE_THR,
    max_graph_nodes_for_hops: int = MAX_GRAPH_NODES_FOR_HOPS) -> dict[str, Any]:
    """Build strict, content-aware metadata for neighbour-cache compatibility checks."""
    node_list = [int(value) for value in node_ids]
    valid_list = [int(value) for value in valid_node_ids] if valid_node_ids is not None else node_list
    spectral_enabled = (
        bool(enable_spectral) and int(spectral_dim) > 0
        and not auto_disable_spectral(int(graph.num_nodes), int(max_graph_nodes_for_hops)))

    return {
        "num_graph_nodes": int(graph.num_nodes),
        "num_center_nodes": len(node_list),
        "center_node_ids_fingerprint": stable_int_fingerprint(node_list),
        "valid_node_ids_fingerprint": stable_int_fingerprint(valid_list),
        "docs_fingerprint": docs_fingerprint(documents),
        "citation_edges_fingerprint": edge_set_fingerprint(graph.edge_set),
        "max_context_size": int(max_context_size),
        "sampling_strategy": str(sampling_strategy),
        "connectivity_weight": float(connectivity_weight),
        "temporal_weight": float(temporal_weight),
        "reciprocity_weight": float(reciprocity_weight),
        "overlap_weight": float(overlap_weight),
        "enable_spectral": bool(spectral_enabled),
        "requested_enable_spectral": bool(enable_spectral),
        "k_hops": int(k_hops),
        "spectral_dim": int(spectral_dim),
        "hub_degree_threshold": int(hub_degree_threshold),
        "max_graph_nodes_for_hops": int(max_graph_nodes_for_hops)}


def neighbor_is_compatible(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return True when all neighbour-cache compatibility keys match exactly."""
    return metadata_matches("neighbor", metadata, expected, COMPATIBILITY_KEYS)
