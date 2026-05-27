# Public module API

[Previous](./07-model-inspection.md) · [Index](./00-index.md) · [Next](./09-results-and-artifacts.md)

---

## API scope

MetaGraphSci exposes Python modules and classes.

## Main classes

| Class | Responsibility |
|---|---|
| `MetaGraphSci` | encode modalities, fuse branches, classify documents, run ablations |
| `TextEncoder` | SciBERT-compatible text representation with optional PEFT |
| `MetadataEncoder` | venue, publisher, author, and year representation |
| `CitationGraphTransformer` | relation-aware citation context encoding |
| `MultimodalFusion` | gated residual fusion of text, metadata, and citation streams |
| `NormalizedCosineClassifier` | prototype-based cosine classifier |
| `NeighborhoodAwareContrastiveLoss` | graph-aware contrastive objective |
| `PseudoLabeler` | confidence-filtered pseudo-label selection |

## Main flow

```python
model = MetaGraphSci(...)
h_text, h_meta, h_citation = model.encode_modalities(...)
logits = model(...)
metrics = evaluate_predictions(y_true, y_pred, y_prob)
save_evaluation_bundle(...)
```

## Probability output

Predicted probabilities should satisfy:

$$
\hat{p}_{i,c}=\frac{\exp(\ell_{i,c})}{\sum_{k=1}^{C}\exp(\ell_{i,k})}, \qquad \sum_{c=1}^{C}\hat{p}_{i,c}=1
$$

---

[Previous](./07-model-inspection.md) · [Index](./00-index.md) · [Next](./09-results-and-artifacts.md)
