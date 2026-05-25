# Configuration

[Previous](./09-results-and-artifacts.md) · [Index](./00-index.md) · [Next](./11-troubleshooting.md)

---

## :gear: YAMLS

The uploaded code does not include a concrete YAML configuration file, but the model constructors expose clear configuration groups.

Use YAML only if the training script is designed to load it. Otherwise, pass the same values through the existing Python training code.

## Suggested configuration groups

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
  peft_target_modules: [query, value]
  gradient_checkpointing: false
  freeze_backbone_until_layer: 0

metadata_encoder:
  metadata_embedding_dim: 64
  metadata_cross_layers: 3

citation_encoder:
  citation_heads: 4
  citation_layers: 2
  citation_ff_dim: 1024
  selector_hidden_dim: 256
  selector_top_k: 8
  max_context_size: 32
  citation_dropout: 0.1
  hop_profile_dim: 2
  spectral_dim: 0
  use_latent_graph: true
  latent_graph_top_k: 4
  hybrid_alpha_init: 0.0

fusion:
  fusion_modality_dropout: 0.1

pseudo_labeler:
  beta: 0.80
  warmup_epochs: 3
  min_per_class: 0
  temperature: 0.5
  ema_momentum: 0.95
  distributionalignment: true
```

## Important notes

- `num_venues`, `num_publishers`, and `num_authors` must match the metadata vocabularies used during preprocessing.
- `num_classes` must match the label mapping.
- `text_dim`, `metadata_dim`, and `citation_dim` must match the encoder outputs expected by fusion.
- `citation_heads` must divide the citation token dimension.
- QLoRA requires compatible GPU/runtime support.
- Pseudo-labeler adaptive state should be saved and restored with checkpoints.
