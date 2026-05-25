# Experiments and evaluation flow

[Previous](./00-index-06-12.md) · [Index](./00-index.md) · [Next](./07-model-inspection.md)

---

## :test_tube: Experiments

MetaGraphSci experiments should focus on classification quality, ablation behavior, representation quality, and numerical stability.

The uploaded code supports these experiment dimensions:

- full multimodal model,
- text-only ablation,
- text plus metadata ablation,
- text plus citation ablation,
- pseudo-labeling behavior,
- evaluation bundle export,
- training-history and diagnostic plots when history rows are available.

## Ablation modes

The available ablation modes are:

```text
full
text_only
text_metadata
text_citation
```

The model computes the modality encoders and then zero-masks the disabled modalities before fusion. This keeps tensor shapes stable while simulating missing modalities.

## Recommended experiment order

1. Run a text-only baseline.
2. Run `text_metadata` to measure the contribution of publication metadata.
3. Run `text_citation` to measure the contribution of citation context.
4. Run `full` to measure the combined multimodal model.
5. Compare validation/test metrics across all modes.
6. Inspect confusion matrices and per-class F1 scores.
7. Inspect calibration if probability outputs are saved.
8. Inspect embedding projections if penultimate embeddings are saved.

## Evaluation outputs

The evaluation utility is designed to collect:

- aggregate multiclass metrics,
- per-class metrics,
- prediction rows,
- confusion plot,
- class-support plot,
- per-class F1 plot,
- confidence histogram,
- optional calibration plot,
- optional embedding projection,
- optional training-history plots,
- optional pseudo-label ratio plot.

## Suggested result comparison table

| Mode | Text | Metadata | Citation | Purpose |
|---|---:|---:|---:|---|
| `text_only` | yes | no | no | Text baseline |
| `text_metadata` | yes | yes | no | Metadata contribution |
| `text_citation` | yes | no | yes | Citation contribution |
| `full` | yes | yes | yes | Complete MetaGraphSci model |

## What not to document here

Do not describe retrieval metrics, RAG traces, Weaviate indexing, Ollama generation, or citation-chip validation unless those components are actually added to the MetaGraphSci codebase.
