# MetaGraphSci documentation index

This folder contains the extended documentation for the MetaGraphSci data and modelling pipeline.

<p align="center">
  <img src="../../images/fusion-part.png" alt="MetaGraphSci architecture overview" width="90%">
</p>

## Reading order

1. [Overview and produced artifacts](./01-overview.md)
2. [Architecture, layers, and formulas](./02-architecture.md)
3. [Local setup and first run](./03-quickstart.md)
4. [Supported datasets](./04-datasets.md)
5. [Cache layer](./05-cache-layer.md)
6. [Experiments and evaluation flow](./06-experiments.md)
7. [Model inspection and debugging](./07-model-inspection.md)
8. [Public module API](./08-public-module-api.md)
9. [Results and artifacts](./09-results-and-artifacts.md)
10. [Configuration](./10-configuration.md)
11. [Troubleshooting](./11-troubleshooting.md)
12. [Roadmap](./12-roadmap.md)

## Pipeline layout

```text
download.py / downloaders.py   -> raw corpora -> normalised CSV bundles
tabular_utils.py               -> documents.csv + label encoding + splits
graph_utils.py / graph_cache   -> citation graph + train/val/test views
tokenization_cache.py          -> per-doc input_ids + attention_mask
embedding_cache.py             -> frozen-encoder document vectors
encoder_cache.py               -> venue / publisher / author vocabularies
context_caching.py             -> ranked neighbour records per center doc
dataset.py                     -> PyTorch Dataset + DataLoader
```

Return to the repository root README: [../../README.md](../../README.md)
