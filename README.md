<div align="center">

<h1>
  MetaGraphSci<br>
  Multimodal Scientific Document Classification
</h1>

<p>
  <b>
  Classify scientific papers by combining text, metadata, and citation-graph context through <br>
  SciBERT encoders, graph-aware citation modeling, gated fusion, and reproducible ablation studies.
  </b>
</p>

<p>
  <a href="#1-system-overview"><img alt="Overview" src="https://img.shields.io/badge/Overview-system%20flow-2ea44f?style=flat-square&logo=readthedocs&logoColor=white"></a>
  <a href="#2-module-reference"><img alt="Modules" src="https://img.shields.io/badge/Modules-encoders%20%7C%20fusion%20%7C%20losses-0969da?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="#3-training-pipeline"><img alt="Training" src="https://img.shields.io/badge/Training-contrastive%20%2B%20semi--supervised-b91c1c?style=flat-square&logo=lightning&logoColor=white"></a>
  <a href="#4-configuration-reference"><img alt="Config" src="https://img.shields.io/badge/Config-YAML%20profiles-8250df?style=flat-square&logo=yaml&logoColor=white"></a>
</p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="SciBERT" src="https://img.shields.io/badge/SciBERT-allenai-6366F1?style=flat-square">
  <img alt="PEFT" src="https://img.shields.io/badge/PEFT-LoRA%20%7C%20QLoRA-22C55E?style=flat-square">
</p>

<p>
  <img alt="Graph" src="https://img.shields.io/badge/Citation%20Graph-relation--aware-0a66c2?style=flat-square&logo=googlescholar&logoColor=white">
  <img alt="Fusion" src="https://img.shields.io/badge/Fusion-gated%20residual-6f42c1?style=flat-square">
  <img alt="Ablations" src="https://img.shields.io/badge/Ablations-text%20%7C%20metadata%20%7C%20citation-d73a49?style=flat-square">
  <img alt="Evaluation" src="https://img.shields.io/badge/Evaluation-macro--F1%20%7C%20plots%20%7C%20artifacts-1f883d?style=flat-square">
</p>

<p>
  <a href="#1-system-overview"><b>✨ Overview</b></a>
  &nbsp;·&nbsp;
  <a href="#2-module-reference"><b>🧩 Modules</b></a>
  &nbsp;·&nbsp;
  <a href="#3-training-pipeline"><b>🚀 Training</b></a>
  &nbsp;·&nbsp;
  <a href="#4-configuration-reference"><b>⚙️ Config</b></a>
  &nbsp;·&nbsp;
  <a href="#6-random-accuracy-debugging-checklist"><b>🛠️ Debugging</b></a>
</p>

</div>

---

## :books: Documentation

The README is organized as a compact technical guide for the current MetaGraphSci codebase.  
It does not document non-existing FastAPI, frontend, RAG, Weaviate, or Ollama components.

| Section | What you will find | Open |
|---|---|---|
| :sparkles: Overview | Project purpose, multimodal signals, and end-to-end data flow. | [Read overview](#1-system-overview) |
| :jigsaw: Module reference | Ablations, SciBERT text encoder, metadata encoder, citation graph encoder, fusion, classifier, losses, and pseudo-labeling. | [View modules](#2-module-reference) |
| :rocket: Training pipeline | Data orchestration, contrastive pretraining, semi-supervised fine-tuning, and checkpoint selection. | [Train model](#3-training-pipeline) |
| :gear: Configuration | Project, cache, data, model, train, and trainer settings. | [Edit config](#4-configuration-reference) |
| :dart: Profiles | Tiny/debug, smoke test, and full experiment configurations. | [Choose profile](#5-recommended-profiles) |
| :wrench: Debugging | Random-accuracy checklist and sanity checks for labels, ablations, pseudo-labeling, and graph context. | [Debug issues](#6-random-accuracy-debugging-checklist) |
| :bar_chart: Sensitivity | High-impact hyperparameters and safe starting values. | [Tune safely](#7-hyperparameter-sensitivity-summary) |

> [!NOTE]
> For a first run, start with the **text-only** ablation. It verifies that labels, tokenization, and the SciBERT classifier can learn before graph and metadata streams are enabled.

---

## 1. System Overview

MetaGraphSci is a multimodal classifier for scientific papers. It jointly models:

- **Text:** title and abstract encoded with SciBERT plus optional LoRA/QLoRA.
- **Metadata:** venue, publisher, authors, and year encoded with a Deep Cross Network.
- **Citation graph:** citation-neighborhood context encoded with a relation-aware graph transformer.

The three representations are fused by a gated residual layer and classified with a normalized cosine-prototype classifier.

---

## 2. Module Reference

### 2.1 Ablation Contracts

The model supports four ablation modes:

```python
ABLATION_MODES = {
    "full": {"text", "metadata", "citation"},
    "text_only": {"text"},
    "text_metadata": {"text", "metadata"},
    "text_citation": {"text", "citation"},
}
```

Ablation runs the full forward pass and zeroes disabled modality tensors before fusion. This keeps tensor shapes fixed across experiments.

`NUM_RELATIONS = 4` corresponds to citation-edge structure, temporal proximity, metadata compatibility, and learned latent adjacency.

### 2.2 TextEncoder

`TextEncoder` wraps a SciBERT-compatible backbone and supports CLS pooling, LoRA, QLoRA, gradient checkpointing, partial layer freezing, and projection to `text_dim`.

LoRA updates small low-rank adapters instead of the full backbone. QLoRA loads the base model in 4-bit quantization to reduce VRAM usage.

### 2.3 MetadataEncoder

`MetadataEncoder` embeds venue, publisher, authors, and year, then applies a Deep Cross Network to model explicit metadata interactions.

Author padding uses `padding_idx=0`, allowing empty author slots to be ignored by masked mean pooling.

### 2.4 CitationGraphTransformer

`CitationGraphTransformer` encodes citation-neighborhood context using:

- `LearnedCitationSelector`,
- `StructuralPEEncoder`,
- `RelationMixer`,
- `GPSCitationLayer`,
- optional `LatentGraphModule`,
- gated pooling.

```text
candidate context nodes
        |
        v
LearnedCitationSelector
        |
        v
GraphTokenizer + StructuralPEEncoder
        |
        v
GPSCitationLayer x N
        |
        v
gated pooling
        |
        v
h_citation
```

The implementation includes stability guards for empty spectral features and all-masked attention rows.

### 2.5 Fusion and Classifier

`MultimodalFusion` concatenates text, metadata, and citation vectors and applies gated residual fusion:

```text
[text, metadata, citation] -> residual projection + gate * nonlinear fusion
```

`NormalizedCosineClassifier` normalizes embeddings and class prototypes, then scales cosine similarities to produce bounded logits.

### 2.6 NeighborhoodAwareContrastiveLoss

`NeighborhoodAwareContrastiveLoss` is a graph-aware InfoNCE objective. It removes known citation neighbors from negatives, downweights metadata-similar negatives, supports graph positives, and includes numerical safeguards against `log(0)`.

### 2.7 PseudoLabeler

`PseudoLabeler` supports semi-supervised learning with distribution alignment, probability sharpening, EMA-based class thresholds, warmup blocking, optional minimum per-class acceptance, and persistent `ema_class_max` state.

### 2.8 MetaGraphSci

`MetaGraphSci` owns the full model:

```text
TextEncoder
MetadataEncoder
CitationGraphTransformer
MultimodalFusion
NormalizedCosineClassifier
```

`encode_modalities()` returns `(h_text, h_meta, h_citation)` after ablation masking.  
`forward()` returns `(embeddings, logits, probabilities)`.  
`get_embeddings()` returns fused embeddings for contrastive pretraining.

---

## 3. Training Pipeline

### 3.1 Data Orchestration

A run builds:

1. documents and citations,
2. label validation,
3. train/validation/test splits,
4. labeled and unlabeled subsets,
5. graph views,
6. tokenizers and metadata encoders,
7. tokenization, embedding, and neighbor caches,
8. pretrain, labeled, unlabeled, validation, and test datasets.

Labels must be contiguous integers from `0` to `num_classes - 1`.

Graph mode can be:

- `transductive`: validation/test nodes are structurally visible, labels withheld,
- `inductive`: each split uses only its own graph context.

### 3.2 Training Stages

**Stage 1: Contrastive pretraining.**  
The model computes anchor embeddings and token-masked positive embeddings. Graph adjacency defines positives, while metadata similarity softens negatives.

**Stage 2: Semi-supervised fine-tuning.**  
Fine-tuning combines supervised cross-entropy with optional pseudo-label loss:

```text
loss = supervised_loss + pseudo_weight(epoch) * pseudo_label_loss
```

Pseudo-labeling starts after supervised warmup and ramps gradually. Best checkpoints are selected on validation metrics such as `macro_f1`.

---

## 4. Configuration Reference

### 4.1 Project

```yaml
project:
  benchmark: "cs_ai"
  run_name: "MetaGraphSci_cs_ai_stable"
  output_dir: "runs/metagraphsci/cs_ai_stable"
  cache_dir: "cache/metagraphsci/cs_ai"
```

### 4.2 Caching

```yaml
caching:
  tokenization_cache: true
  doc_embedding_cache: true
  graph_split_cache: true
  encoder_cache: true
  neighbor_cache: true
```

Invalidate caches when tokenization, graph structure, metadata mappings, context size, sampling strategy, spectral settings, or backbone model changes.

### 4.3 Data

```yaml
data:
  documents: "data/cs_ai/documents.csv"
  citations: "data/cs_ai/citations.csv"
  label_column: "label"
  split_strategy: "time"
  graph_mode: "transductive"
  label_ratio: 0.25
  max_seq_length: 256
  max_context_size: 8
  k_hops: 2
  spectral_dim: 0
  sampling_strategy: "local_relevance"
```

### 4.4 Model

```yaml
model:
  tokenizer_name: "allenai/scibert_scivocab_uncased"
  text_dim: 768
  metadata_dim: 256
  citation_dim: 256
  fusion_dim: 512
  metadata_embedding_dim: 64
  metadata_cross_layers: 2
  classifier_scale: 8.0
  peft_mode: "lora"
  lora_r: 8
  lora_alpha: 16
  gradient_checkpointing: true
  freeze_backbone_until_layer: 6
  citation_heads: 4
  citation_layers: 2
  selector_top_k: 6
  fusion_modality_dropout: 0.15
  use_latent_graph: true
```

### 4.5 Train and Trainer

```yaml
train:
  batch_size: 8
  pretrain_epochs: 3
  finetune_epochs: 25
  seeds: [42, 1337, 2025]
  ablations: ["text_only", "text_metadata", "full"]

trainer:
  mixed_precision: "bf16"
  gradient_accumulation_steps: 4
  pretrain_lr: 1.0e-5
  finetune_lr: 2.0e-5
  weight_decay: 0.01
  max_grad_norm: 1.0
  selection_metric: "macro_f1"
  label_smoothing: 0.05
  lambda_ssl_final: 0.25
  supervised_warmup_epochs: 5
  pseudo_ramp_epochs: 8
```

---

## 5. Recommended Profiles

### Tiny / Debug

```yaml
data:
  label_ratio: 0.30
  max_context_size: 2
  max_seq_length: 128

train:
  ablations: ["text_only"]
  batch_size: 4
  pretrain_epochs: 0
  finetune_epochs: 3

trainer:
  lambda_ssl_final: 0.0
  mixed_precision: "no"

model:
  freeze_backbone_until_layer: 11
  use_latent_graph: false
```

### Smoke Test

```yaml
train:
  ablations: ["text_only", "full"]
  pretrain_epochs: 1
  finetune_epochs: 5

trainer:
  lambda_ssl_final: 0.0
```

### Full Experiment

```yaml
train:
  seeds: [42, 1337, 2025]
  ablations: ["text_only", "text_metadata", "full"]
  pretrain_epochs: 5
  finetune_epochs: 30

trainer:
  lambda_ssl_final: 0.25
  supervised_warmup_epochs: 5
  pseudo_ramp_epochs: 8
```

---

## 6. Random Accuracy Debugging Checklist

If macro F1 is near random:

1. Run `text_only` first.
2. Disable pseudo-labeling.
3. Increase `label_ratio`.
4. Verify contiguous integer labels.
5. Verify every class appears in the labeled split.
6. Overfit 32 examples.
7. Reduce `classifier_scale`.
8. Reduce `max_context_size`.
9. Disable graph modules with `text_only`.
10. Check macro F1, not only accuracy.

---

## 7. Hyperparameter Sensitivity Summary

| Hyperparameter | Risk if too low | Risk if too high | Start |
|---|---|---|---|
| `label_ratio` | Missing class labels | More labeling cost | `0.20-0.25` |
| `max_seq_length` | Truncates abstracts | OOM / slow | `256` |
| `max_context_size` | Weak graph signal | noisy context / OOM | `8` |
| `classifier_scale` | weak gradients | overconfident logits | `8.0` |
| `pretrain_lr` | slow contrastive learning | collapse | `1e-5` |
| `finetune_lr` | slow convergence | forgetting | `2e-5` |
| `lambda_ssl_final` | no SSL benefit | noisy pseudo-labels | `0.15-0.25` |
| `contrastive_temperature` | unstable if very low | weak separation | `0.10` |
| `metadata_cross_layers` | weak interactions | fp16 instability | `2-3` |
| `citation_layers` | shallow graph reasoning | oversmoothing | `2` |
| `fusion_modality_dropout` | modality overreliance | underfitting | `0.10-0.20` |

---

<div align="center">

**Text-aware. Metadata-aware. Citation-aware. Reproducible scientific document classification.**

⭐ Star the repository if this project helps your work.

</div>
