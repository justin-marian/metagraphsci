from enum import IntEnum
from typing import Final, TypeAlias

from torch_geometric.data import Data


# Local Neighborhood Caching vs. Full-Graph Message Passing
# Standard GNNs perform full-batch message passing, which scales poorly and causes 
# "neighbor explosion" on dense scholarly graphs. By defining a `NeighborCache`, 
# the architecture commits to a localized, ego-graph approach. The model will only 
# look at a bounded, pre-computed context of citations per document, allowing it to 
# fit into GPU memory during mini-batch training.
ContextEntry: TypeAlias = dict[str, int | float | list[float]]
NeighborCache: TypeAlias = dict[int, list[ContextEntry]]

# PyG Data Abstraction
# Relying strictly on PyTorch Geometric's `Data` object means the architecture 
# assumes a coordinate format (COO) for sparse graph topology. 
GraphData: TypeAlias = Data


class EdgeType(IntEnum):
    """Categorizes the directionality of citations between documents."""
    # Directed Graph Semantics
    # In scholarly networks, the flow of information is highly directional. 
    # - OUT (Citing): Indicates this paper is building upon older knowledge.
    # - IN (Cited by): Indicates this paper is foundational to newer work.
    # - BIDIRECTIONAL: Often indicates concurrent work or highly intertwined 
    #   subfields. By preserving these as distinct relation types, the RelationMixer 
    #   in the Transformer can learn to weigh historical foundations differently 
    #   from subsequent discoveries.
    NONE = 0
    IN = 1
    OUT = 2
    BIDIRECTIONAL = 3


# Multimodal Schema Enforcement
# MetaGraphSci is a multimodal architecture relying on text, graph, and tabular data.
# By strictly enforcing these columns, we guarantee the downstream model modules 
# (TextEncoder, MetadataEncoder) will always receive the inputs they expect. 
# It forces all heterogeneous academic datasets (Cora, OGBN, FoRC) into a single 
# unified representation space before they ever touch a neural network layer.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "doc_id", 
    "title", 
    "abstract", 
    "venue", 
    "publisher", 
    "authors", 
    "year"
)

DOCUMENT_COLUMNS: Final[tuple[str, ...]] = REQUIRED_COLUMNS + ("label",)

# Safe Embedding Fallbacks
# Neural network embedding tables will crash or produce NaNs 
# if fed null values. Defining global fallbacks ensures that incomplete raw data 
# (e.g., a paper missing its publication year) maps to a distinct, learnable 
# "unknown" vector rather than failing the pipeline.
DEFAULT_YEAR: Final[int] = 2000
UNKNOWN_TOKEN: Final[str] = "<UNK>"
FORC2025_URL: Final[str] = "https://zenodo.org/records/14901529/files/FoRC2025_data.zip?download=1"

# Standardized Evaluation Constraints
# Different graph datasets require fundamentally different validation strategies to prevent data leakage. 
# - 'ogbn_arxiv' uses 'time' splits to ensure the model isn't using future papers 
#   to predict the topics of past papers (temporal leakage).
# - 'cora'/'pubmed' are historically evaluated with 'random' splits.
# Baking this into the pipeline ensures evaluation rigor is strictly tied to the 
# dataset's properties, preventing accidental benchmark cheating.
BENCHMARK_DEFAULTS: Final[dict[str, dict[str, str]]] = {
    "generic": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "random"},
    "cora": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "random"},
    "pubmed": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "random"},
    "ogbn_arxiv": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "time"},
    "forc4cl": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "time"}
}
