from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable, TypeAlias, cast

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.utils import degree, get_laplacian, subgraph as pyg_subgraph
from transformers import AutoTokenizer, PreTrainedTokenizerBase

"""Prepare documents, metadata, and citation context for MetaGraphSci.

This file is the data entry point of the project. It normalizes raw tables,
builds metadata vocabularies, creates a reusable citation-context cache, and
returns dataset items that already contain text, metadata, and bounded citation
context.

- keep every benchmark on one canonical schema,
- move graph preprocessing out of the training loop,
- replace PageRank-based ranking with local relevance selection,
- package the full multimodal sample in one dataset class.
"""

REQUIRED_COLUMNS = ["doc_id", "title", "abstract", "venue", "publisher", "authors", "year"]
DEFAULT_YEAR = 2000
UNKNOWN_TOKEN = "<UNK>"

EDGE_NONE = 0
EDGE_IN = 1
EDGE_OUT = 2
EDGE_BIDIRECTIONAL = 3

ContextEntry: TypeAlias = dict[str, int | float | list[float]]
NeighborCache: TypeAlias = dict[int, list[ContextEntry]]
GraphData: TypeAlias = Data


def parse_authors(value: Any) -> list[str]:
    """Normalize the author field into a clean list."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            return []

    return [part.strip() for part in text.replace("|", ";").split(";") if part.strip()]


def prepare_documents(documents: pd.DataFrame, label_column: str = "label") -> tuple[pd.DataFrame, list[str] | None]:
    """Normalize a raw document table to the schema expected by the pipeline."""
    # Missing columns are a critical error, but we can be forgiving about extra columns and just ignore them.
    df = documents.copy()
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Normalize types and fill missing values with safe defaults. The dataset class will handle the rest.
    df["doc_id"] = cast(pd.Series, pd.to_numeric(df["doc_id"], errors="raise")).astype(int)
    df["title"] = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    df["venue"] = df["venue"].fillna(UNKNOWN_TOKEN).astype(str)
    df["publisher"] = df["publisher"].fillna(UNKNOWN_TOKEN).astype(str)
    df["authors"] = df["authors"].apply(parse_authors)
    df["year"] = cast(pd.Series, pd.to_numeric(df["year"], errors="coerce")).fillna(DEFAULT_YEAR).astype(int)

    # Process labels if present. Support both numeric and string labels, but they must be clean and consistent if they exist.
    # Unlabeled rows will have NaN in the "label" column, which the dataset class can handle.
    label_names: list[str] | None = None
    if label_column in df.columns:
        if pd.api.types.is_numeric_dtype(df[label_column]):
            df["label"] = pd.to_numeric(df[label_column], errors="coerce")
            valid_labels = sorted(df["label"].dropna().astype(int).unique().tolist())
            label_names = [str(label) for label in valid_labels]
        else:
            classes = sorted(df[label_column].dropna().astype(str).unique().tolist())
            mapping = {name: idx for idx, name in enumerate(classes)}
            df["label"] = df[label_column].astype(str).map(mapping)
            label_names = classes

    return df, label_names


def load_documents(path: str | Path, label_column: str = "label") -> tuple[pd.DataFrame, list[str] | None]:
    path = Path(path)
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return prepare_documents(frame, label_column=label_column)


def finalize_graph_data(graph: GraphData) -> GraphData:
    """Attach fast lookup structures to a PyG citation graph."""
    node_ids = graph.node_ids.tolist() if hasattr(graph, "node_ids") else list(range(int(graph.num_nodes)))
    graph.node_id_to_idx = {int(node_id): idx for idx, node_id in enumerate(node_ids)}

    out_neighbors: dict[int, set[int]] = {int(node_id): set() for node_id in node_ids}
    in_neighbors: dict[int, set[int]] = {int(node_id): set() for node_id in node_ids}
    edge_set: set[tuple[int, int]] = set()

    if getattr(graph, "edge_index", None) is not None and graph.edge_index.numel() > 0:
        src = graph.edge_index[0].detach().cpu().tolist()
        dst = graph.edge_index[1].detach().cpu().tolist()
        for src_idx, dst_idx in zip(src, dst):
            src_id = int(node_ids[int(src_idx)])
            dst_id = int(node_ids[int(dst_idx)])

            out_neighbors[src_id].add(dst_id)
            in_neighbors[dst_id].add(src_id)
            edge_set.add((src_id, dst_id))

    graph.out_neighbors = out_neighbors
    graph.in_neighbors = in_neighbors
    graph.edge_set = edge_set
    return graph


def load_citation_graph(path: str | Path, source_col: str = "source", target_col: str = "target", node_ids: Iterable[int] | None = None) -> GraphData:
    """Load the directed citation graph as a torch_geometric Data object."""
    path = Path(path)
    if path.suffix == ".csv":
        edges = pd.read_csv(path)
    elif path.suffix in {".parquet", ".pq"}:
        edges = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if source_col not in edges.columns or target_col not in edges.columns:
        raise ValueError(f"Citation file must contain '{source_col}' and '{target_col}' columns.")

    all_node_ids = sorted(
        set(map(int, node_ids or [])) | 
        set(pd.to_numeric(edges[source_col], errors="raise").astype(int).tolist()) |
        set(pd.to_numeric(edges[target_col], errors="raise").astype(int).tolist()))
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        src_ids = pd.to_numeric(edges[source_col], errors="raise").astype(int).tolist()
        dst_ids = pd.to_numeric(edges[target_col], errors="raise").astype(int).tolist()
        edge_index = torch.tensor([[node_id_to_idx[int(src)] for src in src_ids], [node_id_to_idx[int(dst)] for dst in dst_ids]], dtype=torch.long)

    graph = Data(edge_index=edge_index, num_nodes=len(all_node_ids))
    graph.node_ids = torch.tensor(all_node_ids, dtype=torch.long)
    return finalize_graph_data(graph)


def create_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Instantiate the tokenizer used by the shared text encoder."""
    return AutoTokenizer.from_pretrained(model_name)


def create_encoders(documents: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Build categorical vocabularies for venue, publisher, and authors."""
    venue_encoder = {name: idx + 1 for idx, name in enumerate(sorted(documents["venue"].dropna().astype(str).unique().tolist()))}
    publisher_encoder = {name: idx + 1 for idx, name in enumerate(sorted(documents["publisher"].dropna().astype(str).unique().tolist()))}

    all_authors = sorted({author for values in documents["authors"] if isinstance(values, list) for author in values})

    author_encoder = {name: idx + 1 for idx, name in enumerate(all_authors)}
    for encoder in (venue_encoder, publisher_encoder, author_encoder):
        encoder[UNKNOWN_TOKEN] = 0

    return venue_encoder, publisher_encoder, author_encoder


def split_documents(
    documents: pd.DataFrame, test_size: float, val_size: float,
    seed: int, strategy: str, time_column: str = "year") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train, validation, and test splits."""
    if "label" not in documents.columns or documents["label"].isna().any():
        raise ValueError("split_documents requires a fully labeled dataframe.")

    strategy = strategy.lower()

    if strategy == "time":
        ranked = documents.sort_values([time_column, "doc_id"]).reset_index(drop=True)
        num_total = len(ranked)
        num_test = max(1, int(round(num_total * test_size)))
        num_val = max(1, int(round(num_total * val_size)))

        test_df = ranked.iloc[-num_test:].copy()
        val_df = ranked.iloc[-(num_test + num_val):-num_test].copy()
        train_df = ranked.iloc[:max(0, num_total - num_test - num_val)].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError("Time-based split produced an empty split.")
        return (cast(pd.DataFrame, train_df.reset_index(drop=True)), cast(pd.DataFrame, val_df.reset_index(drop=True)), cast(pd.DataFrame, test_df.reset_index(drop=True)))

    if strategy != "random":
        raise ValueError(f"Unknown split strategy: {strategy}")

    train_df, test_df = train_test_split(documents, test_size=test_size, random_state=seed, stratify=documents["label"])
    val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=seed, stratify=train_df["label"])
    return (cast(pd.DataFrame, train_df.reset_index(drop=True)), cast(pd.DataFrame, val_df.reset_index(drop=True)), cast(pd.DataFrame, test_df.reset_index(drop=True)))


def create_low_label_split(documents: pd.DataFrame, label_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training set into labeled and unlabeled subsets."""
    rng = np.random.default_rng(seed)
    labeled_parts: list[pd.DataFrame] = []
    unlabeled_parts: list[pd.DataFrame] = []

    for label in sorted(documents["label"].dropna().astype(int).unique().tolist()):
        group = documents[documents["label"] == label].reset_index(drop=True)
        order = rng.permutation(len(group))

        keep_count = min(len(group), max(1, int(round(len(group) * label_ratio))))
        labeled_parts.append(group.iloc[order[:keep_count]])
        unlabeled_parts.append(group.iloc[order[keep_count:]])

    labeled = pd.concat(labeled_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    unlabeled = pd.concat(unlabeled_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    unlabeled = unlabeled.copy()
    unlabeled["label"] = np.nan
    return labeled, unlabeled


def subgraph_by_doc_ids(graph: GraphData, doc_ids: Iterable[int]) -> GraphData:
    doc_id_set = set(map(int, doc_ids))
    subset_indices = [graph.node_id_to_idx[node_id] for node_id in graph.node_ids.tolist() if int(node_id) in doc_id_set]
    subset = torch.tensor(subset_indices, dtype=torch.long)
    edge_index, _ = pyg_subgraph(subset, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes, return_edge_mask=True)
    sub = Data(edge_index=edge_index, num_nodes=subset.numel())
    sub.node_ids = graph.node_ids[subset]
    return finalize_graph_data(sub)


def split_graphs(graph: GraphData, train_ids: Iterable[int], val_ids: Iterable[int], test_ids: Iterable[int], mode: str) -> dict[str, GraphData]:
    mode = mode.lower()
    train_set, val_set, test_set = set(map(int, train_ids)), set(map(int, val_ids)), set(map(int, test_ids))

    if mode == "transductive":
        return {"pretrain": finalize_graph_data(graph.clone()), "val": finalize_graph_data(graph.clone()), "test": finalize_graph_data(graph.clone())}
    if mode == "inductive":
        return {"pretrain": subgraph_by_doc_ids(graph, train_set), "val": subgraph_by_doc_ids(graph, val_set), "test": subgraph_by_doc_ids(graph, test_set)}
    if mode == "train_plus_eval":
        return {"pretrain": subgraph_by_doc_ids(graph, train_set), "val": subgraph_by_doc_ids(graph, train_set | val_set), "test": subgraph_by_doc_ids(graph, train_set | test_set)}
    raise ValueError(f"Unknown graph mode: {mode}")


def build_year_lookup(documents: pd.DataFrame) -> dict[int, int]:
    return {int(row["doc_id"]): int(row["year"]) for _, row in documents[["doc_id", "year"]].iterrows()}


def build_local_context_map(graph: GraphData) -> dict[int, set[int]]:
    return {int(node_id): set(graph.out_neighbors.get(int(node_id), set())) | set(graph.in_neighbors.get(int(node_id), set())) - {int(node_id)} for node_id in graph.node_ids.tolist()}


def edge_type(graph: GraphData, center_id: int, neighbor_id: int) -> int:
    incoming = (int(neighbor_id), int(center_id)) in graph.edge_set
    outgoing = (int(center_id), int(neighbor_id)) in graph.edge_set
    if incoming and outgoing:
        return EDGE_BIDIRECTIONAL
    if incoming:
        return EDGE_IN
    if outgoing:
        return EDGE_OUT
    return EDGE_NONE


def reciprocity_value(edge_type: int) -> float:
    return 1.0 if edge_type == EDGE_BIDIRECTIONAL else 0.0


def overlap_score(local_contexts: dict[int, set[int]], node_id: int, neighbor_id: int) -> float:
    node_context = local_contexts.get(node_id, set())
    neighbor_context = local_contexts.get(neighbor_id, set())
    union = node_context | neighbor_context
    if not union:
        return 0.0
    return len(node_context & neighbor_context) / len(union)


def local_relevance_func(
    graph: GraphData, node_ids: Iterable[int],
    documents: pd.DataFrame, connectivity_weight: float,
    temporal_weight: float, reciprocity_weight: float,
    overlap_weight: float) -> dict[int, dict[int, float]]:
    """Rank candidate citations with local structural and temporal relevance."""
    year_lookup = build_year_lookup(documents)
    local_contexts = build_local_context_map(graph)

    in_degree_map = {int(graph.node_ids[idx]): float(value) for idx, value in enumerate(degree(graph.edge_index[1], num_nodes=graph.num_nodes).cpu().tolist())}
    out_degree_map = {int(graph.node_ids[idx]): float(value) for idx, value in enumerate(degree(graph.edge_index[0], num_nodes=graph.num_nodes).cpu().tolist())}
    max_degree = max(max(in_degree_map.values(), default=1.0), max(out_degree_map.values(), default=1.0), 1.0)
    max_degree = float(np.log1p(max_degree))

    relevance: dict[int, dict[int, float]] = {}
    for node_id in map(int, node_ids):
        candidates = local_contexts.get(node_id, set())
        if not candidates:
            relevance[node_id] = {}
            continue

        node_year = year_lookup.get(node_id, np.nan)
        node_scores: dict[int, float] = {}

        for neighbor_id in candidates:
            degree_value = in_degree_map.get(neighbor_id, 0.0) + out_degree_map.get(neighbor_id, 0.0)
            connectivity = np.log1p(degree_value) / max(max_degree, 1e-8)

            neighbor_year = year_lookup.get(neighbor_id)
            if pd.isna(node_year) or pd.isna(neighbor_year):
                temporal = 1.0
            else:
                temporal = float(np.exp(-abs(int(node_year) - int(neighbor_year)) / 5.0))

            edge_type_value = edge_type(graph, node_id, neighbor_id)
            reciprocity = reciprocity_value(edge_type_value)

            overlap = overlap_score(local_contexts, node_id, neighbor_id)
            score = connectivity_weight * connectivity + temporal_weight * temporal + reciprocity_weight * reciprocity + overlap_weight * overlap

            node_scores[neighbor_id] = float(score)
        relevance[node_id] = node_scores
    return relevance


def k_hop_profile(graph: GraphData, node_id: int, max_hops: int) -> list[float]:
    """Count the number of nodes at each hop distance around a node using BFS over PyG adjacency."""
    if max_hops <= 0 or int(node_id) not in graph.node_id_to_idx:
        return []

    undirected = {node: set(graph.out_neighbors.get(node, set())) | set(graph.in_neighbors.get(node, set())) for node in graph.node_id_to_idx.keys()}
    visited = {int(node_id)}
    frontier = {int(node_id)}

    counts = [0.0] * max_hops

    for hop in range(1, max_hops + 1):
        next_frontier: set[int] = set()
        for current in frontier:
            next_frontier.update(undirected.get(current, set()))

        next_frontier -= visited
        counts[hop - 1] = float(len(next_frontier))
        if not next_frontier:
            break

        visited |= next_frontier
        frontier = next_frontier

    total = sum(counts)
    if total > 0:
        counts = [value / total for value in counts]
    return counts


def spectral_features(graph: GraphData, node_ids: Iterable[int], spectral_dim: int, enabled: bool = False, max_nodes: int = 5000) -> dict[int, list[float]]:
    node_list = list(map(int, node_ids))
    if spectral_dim <= 0:
        return {node_id: [] for node_id in node_list}

    if not enabled or len(node_list) > max_nodes:
        return {node_id: [0.0] * spectral_dim for node_id in node_list}

    subset_indices = [graph.node_id_to_idx[node_id] for node_id in node_list if node_id in graph.node_id_to_idx]
    if not subset_indices:
        return {node_id: [0.0] * spectral_dim for node_id in node_list}

    subset = torch.tensor(subset_indices, dtype=torch.long)
    edge_index, _ = pyg_subgraph(subset, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes, return_edge_mask=True)
    if subset.numel() == 0:
        return {node_id: [0.0] * spectral_dim for node_id in node_list}

    undirected_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1) if edge_index.numel() > 0 else edge_index
    lap_edge_index, lap_weight = get_laplacian(undirected_edge_index, normalization="sym", num_nodes=subset.numel())
    lap = torch.zeros((subset.numel(), subset.numel()), dtype=torch.float32)
    if lap_edge_index.numel() > 0:
        lap[lap_edge_index[0], lap_edge_index[1]] = lap_weight.float()

    eigenvalues, eigenvectors = torch.linalg.eigh(lap)
    use_dim = min(spectral_dim, max(0, eigenvectors.size(1) - 1))
    features = torch.zeros((subset.numel(), spectral_dim), dtype=torch.float32)
    if use_dim > 0:
        features[:, :use_dim] = eigenvectors[:, 1:1 + use_dim]

    return {int(graph.node_ids[int(orig_idx)]): features[row_idx].tolist() for row_idx, orig_idx in enumerate(subset.tolist())}


def build_neighbor_cache(
    graph: GraphData, node_ids: Iterable[int],
    documents: pd.DataFrame, max_context_size: int,
    valid_node_ids: Iterable[int] | None = None,
    sampling_strategy: str = "local_relevance",
    connectivity_weight: float = 0.35, temporal_weight: float = 0.35,
    reciprocity_weight: float = 0.15, overlap_weight: float = 0.15,
    k_hops: int = 2, spectral_dim: int = 0, enable_spectral: bool = False) -> NeighborCache:
    """Build the reusable citation-context cache used by the dataset."""

    valid_ids = set(map(int, valid_node_ids)) if valid_node_ids is not None else set(map(int, node_ids))
    node_list = list(map(int, node_ids))
    sampling_strategy = sampling_strategy.lower()

    if sampling_strategy == "local_relevance":
        relevance_scores = local_relevance_func(
            graph, node_list, documents, connectivity_weight=connectivity_weight, 
            temporal_weight=temporal_weight, reciprocity_weight=reciprocity_weight, overlap_weight=overlap_weight)

    elif sampling_strategy == "top_k":
        in_degree_map = {int(graph.node_ids[idx]): float(value) for idx, value in enumerate(degree(graph.edge_index[1], num_nodes=graph.num_nodes).cpu().tolist())}
        out_degree_map = {int(graph.node_ids[idx]): float(value) for idx, value in enumerate(degree(graph.edge_index[0], num_nodes=graph.num_nodes).cpu().tolist())}

        relevance_scores = {}
        for node_id in node_list:
            neighbors = set(graph.out_neighbors.get(node_id, set())) | set(graph.in_neighbors.get(node_id, set()))
            neighbors.discard(node_id)
            relevance_scores[node_id] = {neighbor_id: float(in_degree_map.get(neighbor_id, 0.0) + out_degree_map.get(neighbor_id, 0.0)) for neighbor_id in neighbors}
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    year_lookup = build_year_lookup(documents)
    spectral_lookup = spectral_features(graph, node_list, spectral_dim=spectral_dim, enabled=enable_spectral)
    hop_lookup = {node_id: k_hop_profile(graph, node_id, max_hops=k_hops) for node_id in node_list}

    cache: NeighborCache = {}
    for node_id in node_list:
        ranked = [(neighbor_id, score) for neighbor_id, score in relevance_scores.get(node_id, {}).items() if neighbor_id in valid_ids and neighbor_id != node_id]
        ranked.sort(key=lambda item: item[1], reverse=True)

        node_year = year_lookup.get(node_id)
        entries: list[ContextEntry] = []
        for neighbor_id, score in ranked[:max_context_size]:
            neighbor_year = year_lookup.get(neighbor_id, node_year)
            entries.append({"doc_id": int(neighbor_id), "edge_type": int(edge_type(graph, node_id, neighbor_id)),
                            "year_delta": float(neighbor_year - node_year) if not pd.isna(node_year) and not pd.isna(neighbor_year) else 0.0,
                            "score": float(score), "hop_profile": hop_lookup.get(neighbor_id, [0.0] * k_hops), 
                            "spectral": spectral_lookup.get(neighbor_id, [0.0] * spectral_dim)})
        cache[node_id] = entries
    return cache


def save_neighbor_cache(cache: NeighborCache, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    """Save a citation-context cache to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata or {}, "cache": {str(node_id): [{key: value for key, value in entry.items()} for entry in entries] for node_id, entries in cache.items()}}
    path.write_text(json.dumps(payload, indent=2))


def load_neighbor_cache(path: str | Path) -> tuple[NeighborCache, dict[str, Any]]:
    """Load a previously saved citation-context cache."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")

    payload = json.loads(path.read_text())
    raw_cache = payload.get("cache", {})
    cache: NeighborCache = {}

    for node_id, node_data in raw_cache.items():
        if isinstance(node_data, list):
            entries = [{"doc_id": int(entry.get("doc_id", 0)), "edge_type": int(entry.get("edge_type", EDGE_NONE)), 
                        "year_delta": float(entry.get("year_delta", 0.0)), "score": float(entry.get("score", 0.0)), 
                        "hop_profile": [float(v) for v in entry.get("hop_profile", [])], 
                        "spectral": [float(v) for v in entry.get("spectral", [])]} for entry in node_data]
            cache[int(node_id)] = entries
            continue

        seen: set[int] = set()
        flattened: list[ContextEntry] = []

        for neighbors in node_data.values():
            for neighbor_id in neighbors:
                neighbor_id = int(neighbor_id)
                if neighbor_id in seen:
                    continue
                seen.add(neighbor_id)
                flattened.append({"doc_id": neighbor_id, "edge_type": EDGE_NONE, "year_delta": 0.0, "score": 0.0, "hop_profile": [], "spectral": []})

        cache[int(node_id)] = flattened
    return cache, payload.get("metadata", {})


class MultiScaleDocumentDataset(Dataset):
    """Return one fully prepared multimodal sample per item.

    Despite the historical class name, this dataset now exposes one bounded
    citation context instead of fixed hop buckets. That keeps the training side
    simple while the model decides which candidates matter most.
    """

    def __init__(
        self, documents: pd.DataFrame, tokenizer: PreTrainedTokenizerBase,
        venue_encoder: dict[str, int], publisher_encoder: dict[str, int], author_encoder: dict[str, int],
        max_seq_length: int, max_context_size: int, max_authors: int,
        context_documents: pd.DataFrame | None = None, context_cache: NeighborCache | None = None,
        cache_text: bool = True, pretokenize_context: bool = False,
        hop_profile_dim: int = 2, spectral_dim: int = 0) -> None:

        # The dataset is initialized with the main document table, a tokenizer, categorical encoders for metadata, 
        # and parameters for text processing and context handling.
        self.documents = documents.reset_index(drop=True).copy()
        self.context = (context_documents if context_documents is not None else documents).reset_index(drop=True).copy()

        # The dataset holds the main document table and an optional context document table, w
        # hich may be the same as the main table 
        # or a different one used for citation context.
        self.tokenizer = tokenizer
        self.venue_encoder = venue_encoder
        self.publisher_encoder = publisher_encoder
        self.author_encoder = author_encoder

        # The maximum sequence length for text encoding, the maximum number of context neighbors to include, and the maximum number of authors 
        # to encode are all configurable parameters that control the size and complexity of the input features for each sample.
        self.max_seq_length = int(max_seq_length)
        self.max_context_size = int(max_context_size)
        self.max_authors = int(max_authors)
        self.hop_profile_dim = int(hop_profile_dim)
        self.spectral_dim = int(spectral_dim)

        # The context cache is a precomputed mapping of document IDs to their relevant neighbors in the citation graph, 
        # which allows the dataset to quickly retrieve the citation context for each document without needing to compute it on the fly.
        self.cache_text = bool(cache_text)
        self.context_cache = context_cache or {}
        self.doc_lookup = {int(row[0]): dict(zip(self.context.columns, row[1:])) for row in self.context.itertuples(index=False)}
        self.text_cache: dict[int, dict[str, torch.Tensor]] = {}

        if self.cache_text and pretokenize_context:
            for doc_id, row in self.doc_lookup.items():
                self.text_cache[doc_id] = self.tokenize_encode(title=str(row["title"]), abstract=str(row["abstract"]))

    def __len__(self) -> int:
        return len(self.documents)

    def tokenize_encode(self, title: str, abstract: str) -> dict[str, torch.Tensor]:
        # Tokenize and encode the title and abstract together, using the tokenizer provided during initialization.
        encoded = self.tokenizer(title, abstract, max_length=self.max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": encoded["input_ids"].squeeze(0), "attention_mask": encoded["attention_mask"].squeeze(0)}

    def text_tokenized(self, doc_id: int, title: str, abstract: str) -> dict[str, torch.Tensor]:
        # Retrieve the tokenized text features for a document, using the cache if enabled. 
        # If the document ID is not in the cache, tokenize and encode the title and abstract, 
        # store it in the cache if enabled, and return the features.
        if self.cache_text and doc_id in self.text_cache:
            return self.text_cache[doc_id]
        tokenized = self.tokenize_encode(title=title, abstract=abstract)
        if self.cache_text:
            self.text_cache[doc_id] = tokenized
        return tokenized

    def metadata_struct(self, row: pd.Series) -> dict[str, torch.Tensor]:
        # Encode the categorical metadata fields (venue, publisher, authors) using the provided encoders, 
        # and normalize the year to a [0, 1] range based on a reasonable range of publication years.
        authors = row["authors"][:self.max_authors] if isinstance(row["authors"], list) else []
        author_ids = [self.author_encoder.get(author, 0) for author in authors]  # Use 0 for unknown authors
        author_ids += [0] * max(0, self.max_authors - len(author_ids))  # Pad with zeros if there are fewer authors than max_authors
        normalized_year = (int(row["year"]) - 2000) / 26.0  # Normalize to [0, 1] for years in the range [2000, 2026]
        return {
            "venue_ids": torch.tensor(self.venue_encoder.get(str(row["venue"]), 0), dtype=torch.long),
            "publisher_ids": torch.tensor(self.publisher_encoder.get(str(row["publisher"]), 0), dtype=torch.long),
            "author_ids": torch.tensor(author_ids[:self.max_authors], dtype=torch.long),
            "years": torch.tensor([normalized_year], dtype=torch.float32)
        }

    def empty_neighbor_text(self) -> dict[str, torch.Tensor]:
        # Return a default tokenized representation for neighbors that are missing or invalid, 
        # which consists of zeroed input IDs and attention masks.
        return {
            "input_ids": torch.zeros(self.max_seq_length, dtype=torch.long), 
            "attention_mask": torch.zeros(self.max_seq_length, dtype=torch.long)}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieve a single sample from the dataset, including the document's text features, metadata, and bounded citation context.
        The sample includes the tokenized title and abstract, encoded metadata fields, and a fixed-size context of neighboring
        documents from the citation graph, with their corresponding edge types, year deltas, and relevance scores. 
        If the document has a label, it is included as well.
        """
        row = self.documents.iloc[idx]
        doc_id = int(row["doc_id"])
        # The main document's text features are obtained by tokenizing and encoding the title and abstract, using the cache if enabled.
        text_features = self.text_tokenized(doc_id=doc_id, title=str(row["title"]), abstract=str(row["abstract"]))
        item: dict[str, torch.Tensor] = {"doc_id": torch.tensor(doc_id, dtype=torch.long), **text_features, **self.metadata_struct(row)}

        # The citation context is retrieved from the context cache, which provides a list of neighboring document IDs 
        # along with their edge types, year deltas, and relevance scores.
        center_year = int(row["year"])
        entries = self.context_cache.get(doc_id, [])[:self.max_context_size]
        neighbors, edge_types, year_deltas, scores = [], [], [], []
        hop_profiles, spectral_features = [], []
        neighbor_venue_ids, neighbor_publisher_ids, neighbor_years = [], [], []

        for entry in entries:
            neighbor_id = int(entry["doc_id"])
            if neighbor_id not in self.doc_lookup:
                continue

            neighbor_row = self.doc_lookup[neighbor_id]
            neighbors.append(self.text_tokenized(doc_id=neighbor_id, title=str(neighbor_row["title"]), abstract=str(neighbor_row["abstract"])))
            neighbor_year = int(neighbor_row.get("year", center_year))

            # Clamp neighbor year to [2000, 2026] range, then normalize year delta to [-1, 1]
            clamped_neighbor_year = float(np.clip(neighbor_year, 2000, 2026))
            year_delta = (clamped_neighbor_year - center_year) / 26.0
            edge_types.append(int(entry.get("edge_type", EDGE_NONE)))
            year_deltas.append(year_delta)

            scores.append(float(entry.get("score", 0.0)))
            hop_profiles.append([float(v) for v in entry.get("hop_profile", [])][:self.hop_profile_dim])
            spectral_features.append([float(v) for v in entry.get("spectral", [])][:self.spectral_dim])

            neighbor_venue_ids.append(self.venue_encoder.get(str(neighbor_row.get("venue", UNKNOWN_TOKEN)), 0))
            neighbor_publisher_ids.append(self.publisher_encoder.get(str(neighbor_row.get("publisher", UNKNOWN_TOKEN)), 0))
            neighbor_years.append((float(np.clip(neighbor_year, 2000, 2026)) - 2000.0) / 26.0)

        # If there are fewer valid neighbors than max_context_size, pad the context
        # with empty entries and default values for edge types, year deltas, and scores.
        valid_count = len(neighbors)
        if valid_count < self.max_context_size:
            neighbors.extend([self.empty_neighbor_text() for _ in range(self.max_context_size - valid_count)])
            edge_types.extend([EDGE_NONE] * (self.max_context_size - valid_count))
            year_deltas.extend([0.0] * (self.max_context_size - valid_count))

            scores.extend([0.0] * (self.max_context_size - valid_count))
            hop_profiles.extend([[0.0] * self.hop_profile_dim for _ in range(self.max_context_size - valid_count)])
            spectral_features.extend([[0.0] * self.spectral_dim for _ in range(self.max_context_size - valid_count)])

            neighbor_venue_ids.extend([0] * (self.max_context_size - valid_count))
            neighbor_publisher_ids.extend([0] * (self.max_context_size - valid_count))
            neighbor_years.extend([0.0] * (self.max_context_size - valid_count))

        # The context features are stacked into tensors, and a context mask is created to indicate which entries are valid neighbors versus padding.
        item["context_input_ids"] = torch.stack([neighbor["input_ids"] for neighbor in neighbors])
        item["context_attention_mask"] = torch.stack([neighbor["attention_mask"] for neighbor in neighbors])
        item["context_mask"] = torch.tensor([1] * valid_count + [0] * max(0, self.max_context_size - valid_count), dtype=torch.long)
        item["context_edge_types"] = torch.tensor(edge_types, dtype=torch.long)

        item["context_year_deltas"] = torch.tensor(year_deltas, dtype=torch.float32)
        item["context_scores"] = torch.tensor(scores, dtype=torch.float32)

        item["context_hop_profiles"] = torch.tensor(hop_profiles, dtype=torch.float32) if self.hop_profile_dim > 0 else torch.zeros((self.max_context_size, 0), dtype=torch.float32)
        item["context_spectral"] = torch.tensor(spectral_features, dtype=torch.float32) if self.spectral_dim > 0 else torch.zeros((self.max_context_size, 0), dtype=torch.float32)

        item["context_venue_ids"] = torch.tensor(neighbor_venue_ids, dtype=torch.long)
        item["context_publisher_ids"] = torch.tensor(neighbor_publisher_ids, dtype=torch.long)
        item["context_years"] = torch.tensor(neighbor_years, dtype=torch.float32)

        # If the document has a label, it is included in the item as a tensor. Unlabeled documents will not have this field, 
        # which the model can handle appropriately.
        if "label" in row.index and pd.notna(row["label"]):
            item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    """Create a DataLoader with sensible defaults for this project."""
    kwargs: dict[str, Any] = {
        "dataset": dataset, "batch_size": batch_size,
        "shuffle": shuffle, "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available()}
    if num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(**kwargs)
