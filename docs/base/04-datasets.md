# Datasets: Cora, PubMed, OGBN-Arxiv, FoRC, OpenAlex

[Previous](./03-quickstart.md) · [Index](./00-index.md) · [Next](./05-cache-layer.md)

---

## Supported datasets

Every dataset is normalised to the same layout:

```text
data/<dataset>/
├── documents.csv     doc_id, title, abstract, venue, publisher, authors, year, label
├── citations.csv     source, target
└── config.yaml       paths, split strategy, split sizes
```

| Dataset | Scale | Default split | Main use |
|---|---:|---|---|
| Cora | small | random stratified | first smoke tests |
| PubMed | medium | random stratified | larger Planetoid check |
| OGBN-Arxiv | large | temporal | leakage-safe benchmark |
| FoRC2025 | medium/large | temporal | field-of-research classification |
| OpenAlex | configurable | temporal | custom scientific topics |

## OpenAlex topic example

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

## Split rule

For temporal datasets, train papers should precede validation papers, and validation papers should precede test papers:

$$
\max(y_{train}) \leq \min(y_{val}) \leq \max(y_{val}) \leq \min(y_{test})
$$

> [!CAUTION]
> Avoid random splits for datasets with official chronological protocols, especially OGBN-Arxiv.

---

[Previous](./03-quickstart.md) · [Index](./00-index.md) · [Next](./05-cache-layer.md)
