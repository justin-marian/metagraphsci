# Local setup and first end-to-end run

[Previous](./02-architecture.md) · [Index](./00-index.md) · [Next](./04-datasets.md)

---

## :rocket: Quickstart

This setup downloads one small dataset, builds every cache, and iterates the resulting PyTorch DataLoader. Commands are hidden inside collapsibles so the page stays scannable.

### 1. Check requirements

You need Python 3.10 or newer and Git. A GPU is optional and only used by `embedding_cache`. For OpenAlex downloads, a contact email is recommended (polite pool).

<details>
<summary><b>Show validation commands</b></summary>

```bash
python3 --version
git --version
nvidia-smi || true   # optional, only needed for GPU embedding
```

</details>

> [!WARNING]
> Start with one small dataset (`cora` or `pubmed`). Downloading OpenAlex first means hours of network time before you have validated the rest of the pipeline.

### 2. Clone the repository

<details>
<summary><b>Show clone commands</b></summary>

```bash
git clone https://github.com/justin-marian/metagraphsci
cd metagraphsci
```

</details>

All commands below assume you are in the repository root.

### 3. Create the Python environment

<details>
<summary><b>Show Python environment commands</b></summary>

```bash
python3 -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade packaging tools:

```bash
python3 -m pip install --upgrade pip setuptools wheel
```

</details>

### 4. Install dependencies

<details>
<summary><b>Show installation commands</b></summary>

```bash
pip install -e ".[dev]"
```

If the project does not define a `dev` extra:

```bash
pip install -e .
```

</details>

The pipeline depends on `polars`, `torch`, `torch_geometric`, `transformers`, `scikit-learn`, `numpy`, `joblib`, `loguru`, and `pyyaml`. PyTorch and PyG should be installed with versions matching your CUDA setup before running the cache layer.

### 5. Download one dataset

The download CLI normalises every benchmark to the same on-disk shape: `documents.csv`, `citations.csv`, and `config.yaml`.

<details>
<summary><b>Show dataset download commands</b></summary>

Cora (Planetoid, small):

```bash
python3 -m src.data.download \
  --dataset cora \
  --out_dir data/cora
```

PubMed (Planetoid):

```bash
python3 -m src.data.download \
  --dataset pubmed \
  --out_dir data/pubmed
```

OGBN-Arxiv (OGB):

```bash
python3 -m src.data.download \
  --dataset ogbn_arxiv \
  --out_dir data/ogbn_arxiv
```

FoRC2025 (Zenodo):

```bash
python3 -m src.data.download \
  --dataset forc4cl \
  --out_dir data/forc4cl
```

</details>

> [!IMPORTANT]
> The first run downloads compressed corpora and may take a few minutes. Subsequent runs reuse what is already on disk.

### 6. Inspect the produced files

Before building caches, confirm the schema is what you expect.

<details>
<summary><b>Show inspection commands</b></summary>

```bash
ls data/cora
head -3 data/cora/documents.csv
head -3 data/cora/citations.csv
cat   data/cora/config.yaml
```

</details>

`documents.csv` must contain `doc_id`, `title`, `abstract`, `venue`, `publisher`, `authors`, `year`, and `label`. `citations.csv` must contain `source` and `target` (both `doc_id` values).

### 7. Build all caches

The caches are built lazily by `load_or_build_*` calls. The simplest way to materialise everything is to run a small Python script that touches each cache in order.

<details>
<summary><b>Show cache-build script</b></summary>

```python
# scripts/build_caches.py
import polars as pl
from pathlib import Path

from src.data import (
    build_encoder_cache, build_graph_cache, build_loader,
    build_neighbor_cache, build_tokenization_cache,
    create_tokenizer, load_documents, split_documents)

cache_dir = Path("cache/cora")
cache_dir.mkdir(parents=True, exist_ok=True)

docs, _ = load_documents("data/cora/documents.csv")
train, val, test = split_documents(
    docs, test_size=0.2, val_size=0.1, seed=42, strategy="random")

tok = create_tokenizer("allenai/scibert_scivocab_uncased")
tokens = build_tokenization_cache(docs, tok, max_seq_length=256)
encoders = build_encoder_cache(train)
```

</details>

In practice the training script calls the `load_or_build_*` variants of these functions, which read the cache when valid and rebuild only the changed documents otherwise.

### 8. Iterate one batch

<details>
<summary><b>Show DataLoader smoke test</b></summary>

```python
# scripts/smoke_test.py
from src.data import (
    MultiScaleDocumentDataset, build_loader,
    create_tokenizer, load_documents)

docs, _ = load_documents("data/cora/documents.csv")
tok = create_tokenizer("allenai/scibert_scivocab_uncased")

dataset = MultiScaleDocumentDataset(
    documents=docs, tokenizer=tok,
    context_cache={},      # use a pre-built neighbor cache here
    max_seq_length=256, max_context_size=8)

loader = build_loader(dataset, batch_size=4, shuffle=False)
batch  = next(iter(loader))

print({k: getattr(v, "shape", type(v).__name__) for k, v in batch.items()})
```

</details>

A working smoke test should print a dict where every value has the same first dimension (`batch_size`). If `neighbors` has shape `[batch, context_size, seq_len]` you have a complete sample.

> [!TIP]
> Use the smoke test as your first regression: anything that breaks tensor shapes will show up here long before training.

---

[Previous](./02-architecture.md) · [Index](./00-index.md) · [Next](./04-datasets.md)
