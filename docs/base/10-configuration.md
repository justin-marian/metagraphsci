# Configuration

[Previous](./09-results-and-artifacts.md) · [Index](./00-index.md) · [Next](./11-troubleshooting.md)

---

## Suggested YAML groups

Use YAML only when the training script is designed to load it. Otherwise, pass the same values through Python or CLI arguments.

```yaml
model:
  num_classes: 0
  text_dim: 768
  metadata_dim: 256
  citation_dim: 256
  fusion_dim: 512
  classifier_scale: 20.0
  ablation_mode: full

text_encoder:
  model_name: allenai/scibert_scivocab_uncased
  peft_mode: none
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  gradient_checkpointing: false

metadata_encoder:
  metadata_embedding_dim: 64
  metadata_cross_layers: 3

citation_encoder:
  citation_heads: 4
  citation_layers: 2
  selector_top_k: 8
  max_context_size: 32
  use_latent_graph: true
  latent_graph_top_k: 4

pseudo_labeler:
  beta: 0.80
  warmup_epochs: 3
  temperature: 0.5
  ema_momentum: 0.95
```

## Consistency checks

- `num_classes` must match the label mapping.
- `num_venues`, `num_publishers`, and `num_authors` must match metadata vocabularies.
- `citation_heads` must divide the citation token dimension.
- Pseudo-labeler adaptive state should be saved and restored with checkpoints.

---

[Previous](./09-results-and-artifacts.md) · [Index](./00-index.md) · [Next](./11-troubleshooting.md)
