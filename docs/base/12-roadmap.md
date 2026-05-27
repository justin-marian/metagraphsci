# Roadmap

[Previous](./11-troubleshooting.md) · [Index](./00-index.md)

---

## Next steps

- [x] Add a documented training entry point for supervised and semi-supervised runs.
- [x] Add a small reproducible example that instantiates `MetaGraphSci` with dummy tensors.
- [x] Add unit tests for each ablation mode.
- [x] Add shape tests for text, metadata, citation, fusion, and classifier modules.
- [x] Add numerical stability tests for graph attention masks and spectral features.
- [x] Add checkpoint tests for `PseudoLabeler` adaptive state.
- [x] Add a configuration loader only if the training scripts actually use YAML.
- [x] Add a metrics aggregation script for comparing ablation modes.
- [x] Add clearer experiment folders with saved run configuration and random seed.

## Suggested documentation images

Place these files in `images/` so the README and docs render cleanly:

```text
images/fusion-part.png
images/text-encoder.png
images/metadata-encoder.png
images/graph-encoder.png
images/losses.png
images/EmbeddingClasses.jpg
```

<div align="center">

**MetaGraphSci: text, metadata, and citation-aware scientific document classification.**

</div>

---

[Previous](./11-troubleshooting.md) · [Index](./00-index.md)
