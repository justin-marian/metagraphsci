# Public module API

[Previous](./07-model-inspection.md) · [Index](./00-index.md) · [Next](./09-results-and-artifacts.md)

---

## :zap: API

The uploaded MetaGraphSci code exposes Python modules and classes, not an HTTP API.

## Main model

```python
MetaGraphSci(...)
```

Main responsibilities:

- encode text with a SciBERT-based text encoder,
- encode publication metadata,
- encode citation context,
- fuse text, metadata, and citation streams,
- classify documents using normalized cosine prototypes,
- support modality ablations.

Important methods:

```python
encode_modalities(...)
forward(...)
ablation_study(...)
```

## Text encoder

```python
TextEncoder(...)
```

Responsibilities:

- load a SciBERT-compatible backbone,
- optionally apply LoRA or QLoRA,
- optionally enable gradient checkpointing,
- optionally freeze lower transformer layers,
- return a projected document representation.

## Metadata encoder

```python
MetadataEncoder(...)
```

Responsibilities:

- embed venue IDs,
- embed publisher IDs,
- pool author embeddings,
- encode publication year,
- model explicit feature interactions through a deep cross network.

## Citation graph encoder

```python
CitationGraphTransformer(...)
```

Responsibilities:

- select informative citation-context candidates,
- encode structural position information,
- mix relation-aware attention biases,
- combine global attention with local graph message passing,
- optionally use learned latent adjacency.

## Fusion and classifier heads

```python
MultimodalFusion(...)
NormalizedCosineClassifier(...)
```

Responsibilities:

- fuse text, metadata, and citation streams through gated residual fusion,
- randomly drop metadata/citation streams during training when modality dropout is enabled,
- classify using normalized cosine similarity against learnable class prototypes.

## Losses

```python
NeighborhoodAwareContrastiveLoss(...)
```

Responsibilities:

- normalize embeddings,
- compute temperature-scaled similarity,
- mask known graph neighbors to reduce false negatives,
- soften metadata-related negatives,
- support custom positive masks.

## Pseudo-labeling

```python
PseudoLabeler(...)
```

Responsibilities:

- align predictions to a target prior,
- sharpen probabilities,
- maintain adaptive per-class confidence thresholds,
- select pseudo-labels after warmup,
- save and restore adaptive state.

## Evaluation utilities

```python
evaluate_predictions(...)
save_evaluation_bundle(...)
```

Responsibilities:

- compute aggregate metrics,
- compute per-class metrics,
- save prediction tables,
- export diagnostic plots and optional training-history artifacts.
