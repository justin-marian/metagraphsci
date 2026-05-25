# Architecture, layers, and main components

[Previous](./01-overview.md) · [Index](./00-index.md) · [Next](./03-quickstart.md)

---

## :building_construction: Architecture

The pipeline is split so that each failure can be traced to a specific layer. If batched tensors look wrong, the problem is not automatically "the dataset." It may be label encoding, the citation file, the graph mode, the neighbour scoring weights, the tokenizer, or the frozen encoder.

### Layered view

```text
                  ┌──────────────────────────────────────────────────┐
                  │ download.py + downloaders.py                     │
  raw corpora ──> │ OpenAlex / OGBN-Arxiv / FoRC / Cora / PubMed     │ ──> data/<dataset>/{documents,citations}.csv + config.yaml
                  └──────────────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────────────┐
                  │ tabular_utils.py                                 │
                  │ load_documents, prepare_documents,               │
                  │ split_documents, create_low_label_split          │
                  └──────────────────────────────────────────────────┘
                                       │
                                       ▼
       ┌─────────────────────────────────────────────────────────────┐
       │ Cache layer (each file: build, save, load, load-or-build)   │
       │                                                             │
       │  encoder_cache       venue / publisher / author vocab       │
       │  tokenization_cache  per-doc input_ids + attention_mask     │
       │  embedding_cache     frozen-encoder document vectors        │
       │  graph_cache         edge index + per-stage subgraphs       │
       │  context_caching     ranked neighbour records (centers)     │
       │                                                             │
       │  shared via cache_utils.py:                                 │
       │    docs_fingerprint, per_doc_hashes, edge_set_fingerprint,  │
       │    metadata_matches, write_meta_sidecar, ...                │
       └─────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────────────┐
                  │ dataset.py                                       │
                  │ MultiScaleDocumentDataset + build_loader         │
                  └──────────────────────────────────────────────────┘
                                       │
                                       ▼
                              PyTorch training loop
```

### Main components

| Component | What it does |
|---|---|
| `download.py` | CLI entry point: parses args, validates, prints summary, delegates. |
| `downloaders.py` | Per-source fetch + normalisation (Planetoid, OGBN-Arxiv, FoRC2025, OpenAlex). |
| `download_utils.py` | Schema normalisation, YAML config generation, generic table I/O. |
| `tabular_utils.py` | Document loading, label encoding, train/val/test splits, low-label split. |
| `graph_utils.py` | Edge-list loading, dense remapping, adjacency dicts, subgraph extraction. |
| `cache_utils.py` | Shared hashing, fingerprinting, compatibility checks, sidecar I/O. |
| `tokenization_cache.py` | Incremental tokenisation cache keyed on per-doc content hashes. |
| `embedding_cache.py` | Incremental embedding cache keyed on tokenizer + model settings. |
| `encoder_cache.py` | Vocabulary cache for venue, publisher, and author IDs. |
| `graph_cache.py` | Citation graph + per-stage split views, keyed on edges + split params. |
| `context_caching.py` | Neighbour cache: BFS hops, spectral features, structural scoring. |
| `dataset.py` | PyTorch Dataset that turns one center doc + cached context into tensors. |

The key design choice is to keep raw data, structural features, text features, and final tensors in separate caches. You should be able to rebuild any one of them without rebuilding the others.

> [!TIP]
> Debug from left to right. Documents and labels first, graph second, structural features third, text features fourth, dataset last.

### Cache invalidation in one sentence

Each cache stores a sidecar with a content-aware fingerprint. On load, the fingerprint is compared against what the current run would produce: if every relevant field matches, the cache is reused; otherwise only the changed documents are rebuilt.

---

[Previous](./01-overview.md) · [Index](./00-index.md) · [Next](./03-quickstart.md)
