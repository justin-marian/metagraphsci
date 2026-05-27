"""
Shared constants, types, and lightweight enumerations for the MetaGraphSci data pipeline.

Keeping every magic number in one authoritative module prevents the same value
from drifting independently across loaders, dataset code, scoring functions, and API wrappers.

When a tunable needs to change, e.g. the year-normalisation scale or a default cache weight,
updating it here is sufficient; no other module should embed a raw numeric literal for anything defined below.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Final, TypeAlias

from torch_geometric.data import Data

# Default run-name prefix embedded in auto-generated benchmark YAML configs.
# Change to reflect a different base model or experiment family.
BENCHMARK_RUN_PREFIX: Final[str] = "citescibert"

# Standardized Evaluation Constraints
# Different graph datasets require fundamentally different validation strategies to prevent data leakage.
# - 'ogbn_arxiv' uses 'time' splits to ensure the model isn't using future papers
#   to predict the topics of past papers (temporal leakage).
# - 'cora'/'pubmed' are historically evaluated with 'random' splits.
# Baking this into the pipeline ensures evaluation rigor is strictly tied to the
# dataset's properties, preventing accidental benchmark cheating.
BENCHMARK_DEFAULTS_DATASETS: Final[dict[str, dict[str, str]]] = {
    "generic": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "random"},
    "cora": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "random"},
    "pubmed": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "random"},
    "ogbn_arxiv": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "time"},
    "forc4cl": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "time"},
    "openalex": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "time"}}

FORC2025_URL: Final[str] = "https://zenodo.org/records/14901529/files/FoRC2025_data.zip?download=1"
OGBN_ARXIV_TITLEABS_URL: Final[str] = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
# Raw titles+abstracts aligned 1:1 with PyG Planetoid node indices, published by
# Graph-COM (https://huggingface.co/datasets/Graph-COM/Text-Attributed-Graphs).
# Each .pt is a torch-pickled list[str] in "Title: <t>\n|\\tAbstract: <a>" form.
CORA_RAW_TEXTS_URL: Final[str] = "https://huggingface.co/datasets/Graph-COM/Text-Attributed-Graphs/resolve/main/cora/raw_texts.pt"
PUBMED_RAW_TEXTS_URL: Final[str] = "https://huggingface.co/datasets/Graph-COM/Text-Attributed-Graphs/resolve/main/pubmed/raw_texts.pt"

DATASET_REGISTRY: Final[dict[str, dict[str, str]]] = {
    "cora": {"name": "Cora", "source": "planetoid", "split_strategy": "random"},
    "pubmed": {"name": "PubMed", "source": "planetoid", "split_strategy": "random"},
    "ogbn_arxiv": {"name": "ogbn_arxiv", "source": "ogb", "split_strategy": "time"},
    "forc4cl": {"name": "forc4cl", "source": "zenodo", "split_strategy": "time"},
    "openalex": {"name": "openalex", "source": "openalex", "split_strategy": "time"}}

# GraphData - alias for PyG's Data object. Let type annotations across the codebase
# remain readable without importing PyG everywhere, and makes it easy to swap to a
# different graph container later.
GraphData: TypeAlias = Data

# NeighborCache maps each center document id to a ranked list of neighbor records.
# The list format, rather than a nested dict, preserves insertion order so the training loop
# can slice the top-k entries cheaply. Each record is a plain dict with at minimum:
# doc_id, edge_type, year_delta, score; optionally: hop_profile, spectral.
NeighborCache: TypeAlias = dict[int, list[dict[str, Any]]]

# Number of batches prefetched per DataLoader worker process.
# Only active when num_workers > 0 in building dataloaders; ignored for single-process loading.
DATALOADER_PREFETCH_FACTOR: Final[int] = 2

# Sentinel used in vocabulary encoders for unseen venues, publishers, and authors.
# Index 0 is reserved for this token so padding can share the same bucket without
# colliding with a real entity.
UNKNOWN_TOKEN: Final[str] = "<UNK>"

# Minimum columns a raw document table must contain before the pipeline can proceed.
REQUIRED_COLUMNS: Final[list[str]] = [
    "doc_id", "title", "abstract", "venue", "publisher", "authors", "year"]

# Full ordered schema emitted by the dataset exporter. Extra columns the user
# attaches are appended after this prefix so the loader can always assume a
# stable leading layout.
DOCUMENT_COLUMNS: Final[list[str]] = [
    "doc_id", "title", "abstract", "venue", "publisher", "authors", "year",
    # Document label(s) must be present for training, but the column name(s) can vary.
    "label"]

# Fallback when a paper's publication year is missing or unparseable.
# Chosen to sit well before the training data window so imputed years do not
# accidentally fall inside the test split.
YEAR: Final[int] = 2000

# All raw publication years are shifted by REFERENCE_YEAR then divided by
# YEAR_SCALE before entering the model as floating-point features.
#
# REFERENCE_YEAR anchors the mid-point of the expected year distribution.
# YEAR_SCALE maps a ±26-year window, roughly 1974-2026, into [-1, 1].
#
# Both values are defined here so the dataset layer and the caching layer use
# identical normalisation without re-deriving the arithmetic in two places.
REFERENCE_YEAR: Final[float] = float(YEAR)
YEAR_SCALE: Final[float] = 26.0

# Exponential decay half-life used by the temporal scoring component of the
# neighbour relevance model. A value of 5 means two papers published 5 years
# apart receive a similarity of exp(-1) ~= 0.37; 10 years apart ~= 0.14.
# Increase to de-emphasise recency; decrease to sharpen temporal focus.
TEMPORAL_DECAY_YEARS: Final[float] = 5.0

# Root URL for all OpenAlex REST requests.
OPENALEX_BASE: Final[str] = "https://api.openalex.org"

# Polite-crawler header required by the OpenAlex usage policy.
# Omitting or faking this header may result in rate-limiting or blocking.
OPENALEX_HEADERS: Final[dict[str, str]] = {
    "User-Agent": "MetaGraphSci/0.1 (research dataset builder)"}

# Fields requested from the /works endpoint.
# Limiting the select list significantly reduces payload size for large cursor pages.
# Add fields here if downstream labeling or graph construction needs them.
OPENALEX_SELECT: Final[str] = ",".join([
    "id", "title", "abstract_inverted_index", "authorships", "primary_location",
    "publication_year", "primary_topic", "topics", "concepts", "referenced_works"])

# Maximum works per API page. OpenAlex hard cap is 200.
OPENALEX_PAGE_SIZE: Final[int] = 200

# Seconds to sleep after each page request during cursor-paginated bulk fetches.
# In bigger and lower volumes, the builder can afford a longer sleep to keep
# sustained throughput safely below the anonymous rate limit.
OPENALEX_SLEEP_S: Final[float] = 0.12
OPENALEX_SLOW_SLEEP_S: Final[float] = 0.10

# Seconds to wait before a single automatic retry after a transient failure.
OPENALEX_RETRY_SLEEP_S: Final[float] = 3.0

# HTTP timeout for normal requests.
OPENALEX_TIMEOUT_S: Final[int] = 60

# HTTP timeout for the retry attempt. Longer because the first attempt may have
# hit a transient overload that needs more time to clear.
OPENALEX_RETRY_TIMEOUT_S: Final[int] = 120

# Inter-page sleep used by the dataset builder when running many sequential
# class queries. Slightly more generous because the builder issues bursts
# of requests across multiple label classes.
BUILDER_SLEEP_S: Final[float] = 0.8

# Citation-network expansion
#
# The core problem with query-based dataset construction is low citation
# density: papers are fetched per class-query, so most referenced_works URLs
# point to papers outside the fetched corpus. build_citations then discards
# every cross-set edge, leaving avg degree << 1.
#
# The expansion pass fixes this by:
#   1. Tallying how many in-corpus papers reference each external OpenAlex URL.
#   2. Fetching the top-N most cross-referenced external papers by their IDs.
#   3. Labeling them normally and adding them to the corpus before edge extraction.
#
# This dramatically raises internal citation density because the most-referenced
# external papers are usually foundational / survey works that many papers in
# the corpus already cite.

# A referenced-work URL must be cited by at least this many in-corpus papers
# before it is fetched as an expansion candidate. Raising this value focuses
# the expansion budget on the most central hub papers; lowering it increases
# coverage but may pull in unrelated works.
# Rule of thumb: 2-3 for small corpora (<5k papers), 3-5 for larger ones.
MIN_REF_COUNT: Final[int] = 2

# Hard cap on the number of external papers to fetch during the expansion pass.
# Each expansion paper adds one API batch call and is labeled by the labeler.
# A value of 3000-5000 typically raises avg degree from ~0.5 to 4-8x without
# significantly distorting the class distribution.
MAX_PAPERS: Final[int] = 3_000

# Number of OpenAlex IDs packed into a single filter= request during expansion.
# OpenAlex supports pipe-separated openalex: filters; 50 is a safe ceiling that
# keeps the URL well below server-side length limits while minimising round trips.
BATCH_SIZE: Final[int] = 50

# The 4 weights in the local relevance scoring function control how much
# each structural signal contributes to the final neighbour score. They must
# sum to 1.0; deviating breaks score comparability across datasets.
#
#   Connectivity - rewards high-degree nodes, so well-cited papers rank higher.
#   Temporal     - rewards papers published close in time to the center node.
#   Reciprocity  - rewards mutual citations, meaning bidirectional edges.
#   Overlap      - rewards papers that share many local neighbors.
CONNECTIVITY_WEIGHT: Final[float] = 0.35
TEMPORAL_WEIGHT: Final[float] = 0.35
RECIPROCITY_WEIGHT: Final[float] = 0.15
OVERLAP_WEIGHT: Final[float] = 0.15

# Maximum incoming citations a node may have before being treated as a hub,
# e.g. survey paper or textbook, and excluded from neighbourhood scoring.
# Zero disables hub filtering; set to e.g. 500 to prune extremely dominant nodes.
HUB_DEGREE_THR: Final[int] = 0

# Number of BFS hops used to compute the structural hop-profile feature stored
# alongside each cache entry. 2 is typical for citation datasets; bigger hops
# produce richer profiles but require more expensive traversals during cache construction.
K_HOPS: Final[int] = 2

# Spectral embedding dimension stored per cache entry. Zero disables spectral
# computation by default because eigendecomposition is expensive at scale.
SPECTRAL_DIM: Final[int] = 0

# Eigendecomposition is only attempted on induced subgraphs smaller than this limit.
MAX_NODES_FOR_SPECTRAL: Final[int] = 25_000

# BFS hop profiles are only computed when the graph node count is at or below
# this threshold. Larger graphs produce no hop entries rather than blocking
# the build process with a very long BFS traversal.
MAX_GRAPH_NODES_FOR_HOPS: Final[int] = 25_000

# OpenAlex structured-metadata signal

# Primary-topic display name receives the highest per-item vote because it
# reflects OpenAlex's own direct topic classification for the paper.
PRIMARY_TOPIC_WEIGHT: Final[float] = 2.5

# Subfield of the primary topic is less specific but still reliable.
SUBFIELD_WEIGHT: Final[float] = 1.5

# Field-level mapping entry: even less specific than subfield.
FIELD_WEIGHT: Final[float] = 0.8

# Multiplier applied to the raw OpenAlex topic score for secondary topic hits.
# Downweighted because secondary topics can be noisy or tangential.
SECONDARY_TOPIC_MULTIPLIER: Final[float] = 0.5

# Minimum concept relevance score for a concept to contribute a vote at all.
# Concepts with score < 0.3 are treated as background noise.
CONCEPT_SCORE_THR: Final[float] = 0.3

# Venue signal

# Confidence for an exact venue-name match, the strongest tier of venue evidence.
VENUE_EXACT_WEIGHT: Final[float] = 2.0

# Confidence for a partial / substring venue-name match.
VENUE_PARTIAL_WEIGHT: Final[float] = 1.0

# Global signal merge weights
# Applied after individual signals are accumulated. Higher weight means the
# source is trusted more when labels from different signals disagree.
#
#   Venue:     highest - venue names are an extremely reliable proxy for field.
#   OpenAlex:  high    - OpenAlex topic taxonomy is carefully maintained.
#   LLM:       medium  - GPT-quality reasoning but adds latency and cost.
#   Keywords:  lowest  - surface-form matching, prone to false positives.
WEIGHT_VENUE: Final[float] = 2.0
WEIGHT_OPENALEX: Final[float] = 1.5
WEIGHT_LLM: Final[float] = 1.2
WEIGHT_KEYWORDS: Final[float] = 0.8

# Minimum accumulated weight a source must have on the winning label before
# it is credited as the deciding signal in the label provenance column.
ATTRIBUTION_THR: Final[float] = 1.0

# LLM fallback settings

# Anthropic model used for zero-shot label assignment when all structured
# signals are absent or below the confidence threshold.
MODEL: Final[str] = "claude-sonnet-4-20250514"

# Max tokens expected in the JSON response, containing only label + confidence.
MAX_TOKENS: Final[int] = 64

# Maximum characters of abstract text forwarded to the LLM. Keeps the request
# small while preserving sufficient context for a single-label guess.
ABSTRACT_MAX_CHARS: Final[int] = 800

# Default confidence assumed when the LLM's JSON response omits the field.
CONFIDENCE: Final[float] = 0.7

# Multiplier applied to the LLM's self-reported confidence before merging
# with structured votes. Scales the LLM vote into the same magnitude range.
CONFIDENCE_SCALE: Final[float] = 2.0

# Review routing

# Papers whose highest normalised-probability label falls below this threshold
# are flagged `needs_review=True` rather than accepted automatically. Raise
# to demand higher evidence before accepting a label; lower to accept more.
LABELING_CONFIDENCE_THR: Final[float] = 0.45

# Minimum abstract word count. Papers shorter than this are discarded before
# labeling because they provide insufficient text signal.
MIN_ABSTRACT_LENGTH: Final[int] = 50


class EdgeType(IntEnum):
    """
    Encode citation direction as a compact integer for serialisation.

    Edge metadata is stored in JSON cache files, CSV columns, and tensor features:
    all of which handle integers more naturally than arbitrary string tags.

    NONE          - No citation edge between the two nodes, or a self-loop.
    IN            - The neighbor cites the center node, incoming edge.
    OUT           - The center node cites the neighbor, outgoing edge.
    BIDIRECTIONAL - Both nodes cite each other.
    """

    NONE = 0
    IN = 1
    OUT = 2
    BIDIRECTIONAL = 3


BENCHMARK_DEFAULTS: Final[dict[str, dict[str, Any]]] = {
    "openalex_query": {
        # Time splits reflect real-world deployment: a model trained on
        # historical papers should not observe future publications at train time.
        # Use "random" for stratified splits when temporal order is irrelevant.
        "split_strategy": "time",

        # "transductive" exposes all nodes in every graph view, train/val/test,
        # but masks labels appropriately. "inductive" is stricter: test nodes
        # must never appear in the training graph, which requires more careful
        # edge masking and generally produces lower baseline numbers.
        "graph_mode": "transductive",

        # Fraction of labeled nodes used for the semi-supervised training set.
        # Common ablation values: 0.05, 0.10, 0.20.
        "label_ratio": 0.10, "test_size": 0.20, "val_size": 0.10}}

__all__ = [
    # Type aliases
    "GraphData", "NeighborCache",
    # Document schema
    "YEAR", "DOCUMENT_COLUMNS", "REQUIRED_COLUMNS", "UNKNOWN_TOKEN",
    # Temporal normalisation
    "REFERENCE_YEAR", "TEMPORAL_DECAY_YEARS", "YEAR_SCALE",
    # OpenAlex API
    "BUILDER_SLEEP_S", "OPENALEX_BASE", "OPENALEX_SELECT", "OPENALEX_HEADERS",
    "OPENALEX_PAGE_SIZE", "OPENALEX_RETRY_SLEEP_S", "OPENALEX_RETRY_TIMEOUT_S",
    "OPENALEX_SLEEP_S", "OPENALEX_SLOW_SLEEP_S", "OPENALEX_TIMEOUT_S",
    # Neighbour-cache scoring
    "CONNECTIVITY_WEIGHT", "HUB_DEGREE_THR", "K_HOPS", "OVERLAP_WEIGHT",
    "RECIPROCITY_WEIGHT", "SPECTRAL_DIM", "TEMPORAL_WEIGHT",
    # Graph / spectral limits
    "MAX_GRAPH_NODES_FOR_HOPS", "MAX_NODES_FOR_SPECTRAL",
    # DataLoader
    "DATALOADER_PREFETCH_FACTOR",
    # Labeling signal weights
    "LABELING_CONFIDENCE_THR", "ABSTRACT_MAX_CHARS", "CONFIDENCE_SCALE",
    "CONFIDENCE", "MAX_TOKENS", "MODEL", "WEIGHT_KEYWORDS", "WEIGHT_LLM",
    "WEIGHT_OPENALEX", "WEIGHT_VENUE", "MIN_ABSTRACT_LENGTH", "ATTRIBUTION_THR",
    "CONCEPT_SCORE_THR", "FIELD_WEIGHT", "PRIMARY_TOPIC_WEIGHT",
    "SECONDARY_TOPIC_MULTIPLIER", "SUBFIELD_WEIGHT", "VENUE_EXACT_WEIGHT",
    "VENUE_PARTIAL_WEIGHT",
    # Edge type
    "EdgeType",
    # Citation expansion
    "MIN_REF_COUNT", "MAX_PAPERS", "BATCH_SIZE",
    # Benchmark
    "BENCHMARK_DEFAULTS", "BENCHMARK_DEFAULTS_DATASETS",
    "BENCHMARK_RUN_PREFIX", "DATASET_REGISTRY", "FORC2025_URL", "OGBN_ARXIV_TITLEABS_URL",
    "CORA_RAW_TEXTS_URL", "PUBMED_RAW_TEXTS_URL"]
