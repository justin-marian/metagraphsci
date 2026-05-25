# Cache layer: tokenization, embeddings, encoders, graph, neighbours

[Previous](./04-datasets.md) · [Index](./00-index.md) · [Next](./06-splits-and-graphs.md)

---

## :floppy_disk: Cache layer

The cache layer is the project's main correctness and speed mechanism. Each of the five caches is a content-addressed function of its inputs: a change to a relevant field rebuilds only the affected documents; an irrelevant change is a no-op.

Every cache module follows the same five-function shape:

```text
compute_<name>_metadata(...)      build the signature for the current run
<name>_is_compatible(meta, exp)   compare a saved sidecar to that signature
save_<name>_cache(...)            persist tensors + sidecar
load_<name>_cache(path)           read tensors + sidecar
build_<name>_cache(...)           full rebuild from scratch
load_or_build_<name>_cache(...)   the function you actually call
```

Shared helpers (`docs_fingerprint`, `per_doc_hashes`, `metadata_matches`, `edge_set_fingerprint`, `write_meta_sidecar`, …) live in `cache_utils.py`. Each per-artifact module declares a `COMPATIBILITY_KEYS` tuple at the top, which is the single place to look when you want to know what invalidates that artifact.

### Tokenization cache

| Aspect | Detail |
|---|---|
| Artifact | One `input_ids` tensor + one `attention_mask` tensor per document. |
| File | `tokens.pt` + `tokens.meta.json` |
| Compatibility keys | `tokenizer_name`, `max_seq_length`, `per_doc_hashes` |
| Incremental? | Yes — per-doc hash mismatch rebuilds that doc only. |

The sidecar stores a `{doc_id: sha1}` map. On reload, any doc whose hash matches is reused; any new or changed doc is re-tokenised and the cache is written back. If nothing changed, the file is not rewritten.

<details>
<summary><b>Show tokenization cache usage</b></summary>

```python
from src.data import (
    create_tokenizer, load_documents,
    load_or_build_tokenization_cache)

docs, _ = load_documents("data/cora/documents.csv")
tok = create_tokenizer("allenai/scibert_scivocab_uncased")

tokens, summary = load_or_build_tokenization_cache(
    documents=docs,
    tokenizer_name="allenai/scibert_scivocab_uncased",
    tokenizer=tok,
    max_seq_length=256,
    path="cache/cora/tokens.pt")

print(summary)
# {"reused": 2700, "rebuilt": 0, "removed": 0}
```

</details>

### Embedding cache

| Aspect | Detail |
|---|---|
| Artifact | One dense vector per document, produced by a frozen encoder. |
| File | `embeddings.pt` + `embeddings.meta.json` |
| Compatibility keys | `model_name`, `max_seq_length`, `pooling`, `per_doc_hashes` |
| Incremental? | Yes — per-doc hash mismatch re-encodes that doc only. |

Pooling defaults to `cls`. The encoder is loaded only when there is at least one document to (re)build; otherwise the cache is returned untouched.

<details>
<summary><b>Show embedding cache usage</b></summary>

```python
from src.data import load_or_build_embedding_cache

embeddings, doc_ids, summary = load_or_build_embedding_cache(
    documents=docs,
    model_name="allenai/scibert_scivocab_uncased",
    max_seq_length=256,
    tokenizer=tok,
    path="cache/cora/embeddings.pt",
    tokenized_lookup=tokens,
    batch_size=32, device="cuda", pooling="cls")
```

</details>

> [!TIP]
> Pass `tokenized_lookup=tokens` to skip re-tokenising during rebuilds. The embedding cache reuses the tokens cache when all requested doc ids are present.

### Encoder cache

| Aspect | Detail |
|---|---|
| Artifact | Three 1-indexed vocabularies: venue, publisher, author. `<UNK>` maps to 0. |
| File | `encoders.json` |
| Compatibility keys | `seed`, `num_train_docs`, `train_docs_fingerprint` |
| Incremental? | No — vocabularies are small; a full rebuild is fast. |

The encoder cache is built from the **training-split documents only**, so labels from val/test never leak into the metadata embedding tables. The JSON is human-readable so vocabulary drift across runs can be diffed by hand.

<details>
<summary><b>Show encoder cache usage</b></summary>

```python
from src.data import build_encoder_cache, save_encoder_cache, compute_encoder_metadata

encoders = build_encoder_cache(train_docs)
metadata = compute_encoder_metadata(train_docs, seed=42)
save_encoder_cache(encoders, "cache/cora/encoders.json", metadata)
```

</details>

### Graph cache

| Aspect | Detail |
|---|---|
| Artifact | Full citation graph + per-stage views (`pretrain`, `val`, `test`). |
| File | `graph.pt` |
| Compatibility keys | `seed`, `graph_mode`, `split_strategy`, `test_size`, `val_size`, `source_col`, `target_col`, `citations_path`, `documents_path`, `docs_fingerprint`, `citation_edges_fingerprint` |
| Incremental? | No — the graph is built atomically. Edge or document changes trigger a full rebuild. |

The citation-edges fingerprint is the slowest of the cache keys for large corpora. The pipeline memoises it process-locally by `(path, mtime_ns, size)` so repeated validations within one run read the citations file at most once.

<details>
<summary><b>Show graph cache usage</b></summary>

```python
from src.data import (
    build_graph_cache, compute_graph_metadata,
    graph_is_compatible, load_graph_cache, save_graph_cache)

data_cfg = {
    "citations": "data/cora/citations.csv",
    "documents": "data/cora/documents.csv",
    "source_col": "source", "target_col": "target",
    "graph_mode": "transductive", "split_strategy": "random",
    "test_size": 0.2, "val_size": 0.1}

full_graph, splits = build_graph_cache(data_cfg, docs, train_ids, val_ids, test_ids)
metadata = compute_graph_metadata(data_cfg, docs, seed=42)
save_graph_cache(full_graph, splits, "cache/cora/graph.pt", metadata)
```

</details>

### Neighbour cache

| Aspect | Detail |
|---|---|
| Artifact | Ranked context records per center document. |
| File | `neighbors.json` |
| Record fields | `doc_id`, `edge_type`, `year_delta`, `score`, optional `hop_profile`, optional `spectral` |
| Incremental? | No — built once per split and reused. |

Each center document gets up to `max_context_size` ranked neighbours, scored by a weighted combination of:

- **connectivity** — neighbour's normalised total degree
- **temporal similarity** — exponential decay by year distance
- **reciprocity** — bidirectional edges score 1.0
- **overlap** — Jaccard of one-hop neighbourhoods

Scoring runs over multiple processes via `joblib.loky` workers. The output is deterministic across worker counts because chunks are contiguous and merged in submission order.

<details>
<summary><b>Show neighbour cache usage</b></summary>

```python
from src.data import build_neighbor_cache, save_neighbor_cache

cache = build_neighbor_cache(
    graph=full_graph,
    node_ids=train_ids,
    documents=docs,
    max_context_size=8,
    valid_node_ids=train_ids,    # inductive: restrict neighbours to train
    sampling_strategy="local_relevance",
    connectivity_weight=0.35, temporal_weight=0.35,
    reciprocity_weight=0.15, overlap_weight=0.15,
    enable_spectral=False, k_hops=2,
    n_jobs=-1)

save_neighbor_cache(cache, "cache/cora/neighbors.json")
```

</details>

> [!NOTE]
> The neighbour cache is saved as **indented JSON** on purpose. Quality often needs manual inspection during ablation work and the file is small enough that the formatting overhead is acceptable.

### Cache invalidation summary

The full picture in one table:

| Change | Tokens | Embeddings | Encoders | Graph | Neighbours |
|---|---|---|---|---|---|
| One doc's `title` or `abstract` | rebuild that doc | rebuild that doc | -- | -- | -- |
| One doc's `venue` / `publisher` / `authors` | rebuild that doc | rebuild that doc | rebuild | -- | -- |
| One doc's `year` | rebuild that doc | rebuild that doc | -- | -- | rebuild |
| One doc's `label` | rebuild that doc | rebuild that doc | -- | -- | -- |
| Add or remove a citation edge | -- | -- | -- | rebuild | rebuild |
| Change `tokenizer_name` / `max_seq_length` | full rebuild | full rebuild | -- | -- | -- |
| Change `model_name` / `pooling` | -- | full rebuild | -- | -- | -- |
| Change `seed` / split params | -- | -- | rebuild | rebuild | rebuild |
| Change `graph_mode` | -- | -- | -- | rebuild | rebuild |
| Change neighbour scoring weights | -- | -- | -- | -- | rebuild |

> [!CAUTION]
> Hand-editing a `*.meta.json` sidecar to silence an invalidation produces silently wrong results. Either rebuild or accept the staleness; never patch the sidecar.

---

[Previous](./04-datasets.md) · [Index](./00-index.md) · [Next](./06-splits-and-graphs.md)
