# Model inspection and debugging

[Previous](./06-experiments.md) · [Index](./00-index.md) · [Next](./08-public-module-api.md)

---

## What to inspect

Useful inspection points are text encoder output, metadata encoder output, citation encoder output, ablated tensors, fused embedding, classifier logits, probabilities, pseudo-label thresholds, and exported evaluation plots.

<p align="center">
  <img src="../../images/EmbeddingClasses.jpg" alt="Embedding class clusters" width="95%">
</p>

## Forward-pass inspection

```python
h_text, h_meta, h_citation = model.encode_modalities(...)
fused, logits, probs = model(...)
```

If `return_parts=True` is supported locally, use it to inspect intermediate representations.

## Ablation sanity check

```text
text_only      -> metadata = 0, citation = 0
text_metadata  -> citation = 0
text_citation  -> metadata = 0
full           -> no modality is zeroed
```

## Logit check

For a batch with \(B\) samples and \(C\) classes, logits should have shape:

$$
\ell \in \mathbb{R}^{B \times C}
$$

---

[Previous](./06-experiments.md) · [Index](./00-index.md) · [Next](./08-public-module-api.md)
