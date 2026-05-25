# Datasets: Cora, PubMed, OGBN-Arxiv, FoRC, OpenAlex

[Previous](./03-quickstart.md) · [Index](./00-index.md) · [Next](./05-cache-layer.md)

---

## :books: Supported datasets

Every dataset is normalised to the same on-disk shape so the rest of the pipeline does not care where the data came from:

```text
data/<dataset>/
├── documents.csv     doc_id, title, abstract, venue, publisher,
│                     authors, year, label
├── citations.csv     source, target (both doc_id values)
└── config.yaml       paths, split strategy, and split sizes
```

The CLI front-end is `download.py`. Each dataset has its own normalisation rules but the produced files are interchangeable.

### Cora (Planetoid)

The historical citation-classification baseline. Seven classes, ~2.7k documents. Use it for fast iteration: every cache builds in seconds.

<details>
<summary><b>Show Cora command</b></summary>

```bash
python3 -m src.data.download --dataset cora --out_dir data/cora
```

</details>

Default split strategy: `random` (stratified).

### PubMed (Planetoid)

Three classes, ~19k documents. Slightly bigger than Cora, same split strategy.

<details>
<summary><b>Show PubMed command</b></summary>

```bash
python3 -m src.data.download --dataset pubmed --out_dir data/pubmed
```

</details>

Default split strategy: `random` (stratified).

### OGBN-Arxiv (OGB)

Subject-area classification on arXiv CS papers. ~170k documents. The dataset ships with a chronological split that the pipeline preserves to prevent temporal leakage.

<details>
<summary><b>Show OGBN-Arxiv command</b></summary>

```bash
python3 -m src.data.download --dataset ogbn_arxiv --out_dir data/ogbn_arxiv
```

</details>

Default split strategy: `time`. Training papers are strictly older than validation, which are strictly older than test.

> [!CAUTION]
> Do not switch OGBN-Arxiv to a random split. Test labels would leak into training because OGBN's qrels were built assuming a temporal split.

### FoRC2025 (Zenodo)

Field-of-Research classification corpus from Zenodo. The downloader handles zip extraction, document normalisation, and citation extraction.

<details>
<summary><b>Show FoRC command</b></summary>

```bash
python3 -m src.data.download --dataset forc4cl --out_dir data/forc4cl
```

</details>

Default split strategy: `time`.

### OpenAlex (live API)

The flexible option. You supply a topic filter and a max work count; the downloader pages through the OpenAlex `/works` API, normalises abstracts (inverted-index reconstruction), extracts citations, and writes the same on-disk shape.

<details>
<summary><b>Show OpenAlex command</b></summary>

```bash
python3 -m src.data.download \
  --dataset openalex \
  --out_dir data/openalex_cs \
  --oa_filter "primary_topic.field.id:17" \
  --oa_max_works 50000 \
  --oa_email "you@example.org" \
  --oa_label_field field \
  --oa_workers 4 \
  --oa_year_min 2000 \
  --oa_year_max 2024
```

</details>

Useful filter expressions:

```text
primary_topic.field.id:17          Computer Science
primary_topic.subfield.id:1702     Artificial Intelligence
primary_topic.id:T10320            a specific topic
```

Workers > 1 partition the `[year_min, year_max]` range into disjoint year buckets, which OpenAlex serves in parallel without rate-limit collisions.

> [!IMPORTANT]
> OpenAlex filters topics by ID, not by display name. Looking up the right ID once in the OpenAlex topic browser saves a lot of time.

### Default split strategies

The defaults are baked into the generated `config.yaml`:

| Dataset | Split | Reason |
|---|---|---|
| cora | random | Historical baseline; no temporal labels available. |
| pubmed | random | Same as Cora. |
| ogbn_arxiv | time | Official OGB protocol; prevents leakage. |
| forc4cl | time | Future field labels must not leak backwards. |
| openalex | time | Year is reliable; topics drift over time. |

You can override the strategy at training time, but the dataset-shipped default is the safe choice for benchmarking.

---

[Previous](./03-quickstart.md) · [Index](./00-index.md) · [Next](./05-cache-layer.md)
