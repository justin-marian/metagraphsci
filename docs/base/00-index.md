# MetaGraphSci data pipeline documentation

This folder contains the full README content split into smaller local Markdown files.

## Reading order

- [Overview and what the pipeline produces](./01-overview.md)
- [Architecture, layers, and main components](./02-architecture.md)
- [Local setup and first end-to-end run](./03-quickstart.md)
- [Datasets: Cora, PubMed, OGBN-Arxiv, FoRC, OpenAlex](./04-datasets.md)
- [Cache layer: tokenization, embeddings, encoders, graph, neighbours](./05-cache-layer.md)
- [Splits and graph modes (transductive, inductive, train+eval)](./06-splits-and-graphs.md)
- [Dataset and DataLoader integration](./07-dataset-loader.md)
- [Public API reference](./08-api.md)
- [Expected outputs and run context](./09-results-and-artifacts.md)
- [YAML configuration files](./10-configuration.md)
- [Common runtime and build issues](./11-troubleshooting.md)
- [Planned improvements](./12-roadmap.md)

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
