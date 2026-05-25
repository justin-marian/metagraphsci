# Overview and what the pipeline produces

[Index](./00-index.md) · [Index](./00-index.md) · [Next](./02-architecture.md)

---

## :sparkles: Overview

MetaGraphSci is a data pipeline for citation-aware document classification. It turns raw academic corpora (OpenAlex, OGBN-Arxiv, FoRC, Cora, PubMed) into a fixed-shape PyTorch dataset that pairs each document with a ranked, structurally scored citation neighbourhood.

Most graph-classification pipelines hide the preprocessing inside one large script and rebuild everything on every run. This pipeline does the opposite: each stage is a separate cache with content-aware invalidation. A change to a document's abstract invalidates only that document's tokens and embeddings; a change to a citation edge invalidates only the graph and neighbour caches.

The stack is local by default:

- Polars handles tabular loading, normalisation, and splits.
- PyTorch Geometric stores the citation graph in dense tensor form, with adjacency dicts attached for O(1) neighbour lookups.
- HuggingFace tokenizers and frozen encoders produce text features.
- Cached per-document hashes drive incremental rebuilds.
- A `MultiScaleDocumentDataset` exposes ready-to-batch tensors to any PyTorch training loop.

> [!NOTE]
> This is a preprocessing layer, not a model. It produces the inputs that a downstream classifier consumes. The training code lives elsewhere.

A normal development pass looks something like this:

1. Download one dataset, usually `cora` first.
2. Inspect `documents.csv` and `citations.csv` to confirm the schema.
3. Build encoders, tokens, embeddings, and the citation graph (each cached).
4. Build the neighbour cache for the train split.
5. Wrap everything in `MultiScaleDocumentDataset` and iterate a few batches.
6. Move to a larger dataset (OpenAlex or OGBN-Arxiv) only after the small one works.

### Why the project exists

Reproducible graph experiments are slow when every run re-tokenises 50k abstracts, re-runs the frozen encoder, re-reads citations, and re-scores neighbours. They are also fragile when the cache key is "the dataset name": changing a single document doesn't invalidate the cache, so stale tensors silently corrupt results.

This pipeline takes a different position. Every cache file ships a content fingerprint computed from the rows that actually affected it. If those rows change, the cache rebuilds the changed documents only. If they don't change, the cache loads from disk in under a second.

---

## :package: What is included

The repository combines a CLI for dataset construction with cache modules consumed by training scripts. The same caches are used by experiment runs, ablation sweeps, and notebook exploration.

**Download path.** Raw corpora are fetched from their original sources, normalised to the project schema (`doc_id`, `title`, `abstract`, `venue`, `publisher`, `authors`, `year`, `label`), and written as `documents.csv` + `citations.csv` next to a generated `config.yaml`.

**Tabular path.** Documents are loaded, label-encoded, and split into train/val/test using either stratified random or temporal strategies.

**Graph path.** Citation edges are remapped from external `doc_id` to dense tensor indices, stored as a PyG `Data` object, and split into per-stage views (`pretrain`, `val`, `test`).

**Cache path.** Five incremental caches sit between raw data and the training loop:

```text
tokens        per-doc input_ids + attention_mask
embeddings    frozen-encoder vectors (one row per doc)
encoders      venue / publisher / author vocabularies
graph         citation edge index + train/val/test split views
neighbours    ranked context records per center document
```

**Dataset path.** `MultiScaleDocumentDataset` reads the neighbour cache, tokenises center text on demand, and returns fixed-shape tensors that any PyTorch training loop can batch.

A good first local configuration is:

```text
dataset       = cora
split         = random (stratified)
graph_mode    = transductive
tokenizer     = allenai/scibert_scivocab_uncased
max_seq_len   = 256
context_size  = 8
```

> [!IMPORTANT]
> Treat these as starting values, not universal defaults. The authoritative source is the YAML written next to each dataset in `data/<dataset>/config.yaml`.

---

[Index](./00-index.md) · [Index](./00-index.md) · [Next](./02-architecture.md)