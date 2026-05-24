# Model inspection and debugging

[Previous](./06-experiments.md) · [Index](./00-index.md) · [Next](./08-public-module-api.md)

---

## :mag: Model inspection

The uploaded MetaGraphSci code does not include a web playground or frontend.  
This section therefore documents model-level inspection only.

## What to inspect

Useful inspection points are:

1. Text encoder output.
2. Metadata encoder output.
3. Citation encoder output.
4. Ablated modality tensors.
5. Fused multimodal embedding.
6. Classifier logits and probabilities.
7. Pseudo-label confidence thresholds.
8. Evaluation plots and prediction tables.

## Forward-pass inspection

For debugging, inspect the output of:

```python
h_text, h_meta, h_citation = model.encode_modalities(...)
```

Then compare:

```python
fused, logits, probs = model(...)
```

If `return_parts=True` is supported in the local version, use it to inspect intermediate modality representations.

## Ablation inspection

Ablation should verify that the correct modalities are zeroed:

```text
text_only      -> metadata=0, citation=0
text_metadata  -> citation=0
text_citation  -> metadata=0
full           -> no modality is zeroed
```

## Pseudo-label inspection

For semi-supervised runs, inspect:

- adjusted probabilities,
- pseudo-labels,
- confidence values,
- per-class adaptive thresholds,
- kept/rejected pseudo-label mask.

## Not present in current code

The uploaded code does not show:

- a browser-based playground,
- FastAPI endpoints,
- interactive prompt viewer,
- RAG-mode UI,
- citation chips.

Those should not be documented as existing features unless their code is added.
