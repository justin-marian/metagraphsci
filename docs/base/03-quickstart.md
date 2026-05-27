# Local setup and first end-to-end run

[Previous](./02-architecture.md) · [Index](./00-index.md) · [Next](./04-datasets.md)

---

## Quickstart

This setup downloads one small dataset, builds caches, and iterates a PyTorch DataLoader.

### 1. Check requirements

```bash
python3 --version
git --version
nvidia-smi || true
```

You need Python 3.10 or newer. A GPU is optional for small smoke tests, but useful for embedding and full training.

### 2. Create the environment

```bash
git clone https://github.com/justin-marian/metagraphsci
cd metagraphsci
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]" || pip install -e .
```

### 3. Download one small dataset

```bash
python3 -m src.data.download \
  --dataset cora \
  --out_dir data/cora
```

> [!WARNING]
> Start with `cora` or `pubmed`. Downloading OpenAlex first can hide simple schema errors behind a long network job.

### 4. Inspect the bundle

```bash
ls data/cora
head -3 data/cora/documents.csv
head -3 data/cora/citations.csv
cat data/cora/config.yaml
```

The document table should contain `doc_id`, `title`, `abstract`, `venue`, `publisher`, `authors`, `year`, and `label`. The citation table should contain `source` and `target` document IDs.

### 5. Smoke-test a batch

```python
from src.data import MultiScaleDocumentDataset, build_loader, create_tokenizer, load_documents

docs, _ = load_documents("data/cora/documents.csv")
tok = create_tokenizer("allenai/scibert_scivocab_uncased")

dataset = MultiScaleDocumentDataset(
    documents=docs,
    tokenizer=tok,
    context_cache={},
    max_seq_length=256,
    max_context_size=8,
)
loader = build_loader(dataset, batch_size=4, shuffle=False)
batch = next(iter(loader))
print({k: getattr(v, "shape", type(v).__name__) for k, v in batch.items()})
```

A valid batch has a consistent first dimension equal to `batch_size`.

---

[Previous](./02-architecture.md) · [Index](./00-index.md) · [Next](./04-datasets.md)
