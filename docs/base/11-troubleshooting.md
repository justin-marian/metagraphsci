# Troubleshooting

[Previous](./10-configuration.md) · [Index](./00-index.md) · [Next](./12-roadmap.md)

---

## Common checks

| Symptom | Check |
|---|---|
| SciBERT loading fails | model name, dependencies, local cache, network access |
| QLoRA fails | CUDA, bitsandbytes, GPU support, dtype compatibility |
| Fusion shape mismatch | `text_dim`, `metadata_dim`, `citation_dim`, `fusion_dim` |
| Invalid ablation mode | must be `full`, `text_only`, `text_metadata`, or `text_citation` |
| NaN in graph encoder | empty context rows, all-masked attention, invalid adjacency masks |
| NaN in metadata encoder | invalid year values, extreme metadata magnitudes, empty author lists |
| Too few pseudo-labels | warmup, beta, temperature, class imbalance, restored EMA state |
| Missing evaluation files | required predictions, probabilities, embeddings, or history rows absent |

## Attention-mask safety

A common NaN source is a row where all context tokens are masked. Ensure every attention row has at least one valid token:

$$
\sum_{j=1}^{K} M_{i,j} \geq 1 \quad \forall i
$$

---

[Previous](./10-configuration.md) · [Index](./00-index.md) · [Next](./12-roadmap.md)
