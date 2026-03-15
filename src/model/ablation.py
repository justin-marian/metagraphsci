from typing import Final, TypeAlias
from torch import Tensor

# Multimodal Output Standardization
# By strictly defining the output boundary of modality encoders as a 3-tuple 
# of Tensors (Text, Metadata, Citation), guarantee that the downstream Fusion 
# and Classification modules always receive a consistent, predictable payload, 
# regardless of which modalities are currently active or ablated.
TensorTriplet: TypeAlias = tuple[Tensor, Tensor, Tensor]

# Dynamic Ablation Toggles (guarantee immutability)
# During the forward pass, the model checks this mapped set. If a modality is missing 
# (e.g., "metadata" in "text_citation" mode), it zeroes out that specific tensor 
# just before the multimodal fusion step.
ABLATION_MODES: Final[dict[str, frozenset[str]]] = {
    "full": frozenset({"text", "metadata", "citation"}),
    "text_only": frozenset({"text"}),
    "text_metadata": frozenset({"text", "metadata"}),
    "text_citation": frozenset({"text", "citation"}),
}

# Multi-Relational Graph Bias
# RelationMixer doesn't just use binary adjacency edges. It learns a dynamically 
# weighted combination of 4 distinct structural signals to bias the Transformer's attention:
# 1. Structural Citation Edges (In/Out/Bidirectional)
# 2. Temporal Proximity (Year Deltas)
# 3. Metadata Compatibility (Venue/Publisher matches)
# 4. Learned Latent Adjacency (Discovering same-topic but un-cited bridges)
NUM_RELATIONS: Final[int] = 4
