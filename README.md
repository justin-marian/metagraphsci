# MetaGraphSci

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Accelerate-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/PEFT-LoRA%20%7C%20QLoRA-22C55E?style=flat-square" />
  <img src="https://img.shields.io/badge/SciBERT-allenai-6366F1?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

This document is a comprehensive technical reference for the MetaGraphSci codebase: a multimodal deep learning pipeline for scientific document classification. It covers the full system, architecture, modules, training stages, data pipeline, caching, and all YAML configuration hyperparameters in enough detail to understand, debug, and extend every component.

---

## Table of Contents

- [MetaGraphSci](#metagraphsci)
  - [Table of Contents](#table-of-contents)
  - [1. System Overview 🔭](#1-system-overview-)
    - [Data flow](#data-flow)
  - [2. Module Reference 🧩](#2-module-reference-)
    - [2.1 Ablation Contracts](#21-ablation-contracts)
    - [2.2 TextEncoder 📖](#22-textencoder-)
    - [2.3 MetadataEncoder 🏛️](#23-metadataencoder-️)
    - [2.4 CitationGraphTransformer 🕸️](#24-citationgraphtransformer-️)
      - [Sub-modules](#sub-modules)
    - [2.5 MultimodalFusion and NormalizedCosineClassifier 🔀](#25-multimodalfusion-and-normalizedcosineclassifier-)
      - [MultimodalFusion](#multimodalfusion)
      - [NormalizedCosineClassifier](#normalizedcosineclassifier)
    - [2.6 NeighborhoodAwareContrastiveLoss 📉](#26-neighborhoodawarecontrastiveloss-)
    - [2.7 PseudoLabeler 🏷️](#27-pseudolabeler-️)
    - [2.8 MetaGraphSci Top-Level Model 🏗️](#28-metagraphsci-top-level-model-️)
  - [3. Training Pipeline 🚀](#3-training-pipeline-)
    - [3.1 Data Orchestration](#31-data-orchestration)
    - [3.2 MetaGraphSciTrainerEval ⚙️](#32-metagraphscitrainereval-️)
      - [Stage 1 - Contrastive Pretraining](#stage-1---contrastive-pretraining)
      - [Stage 2 - Semi-Supervised Fine-Tuning](#stage-2---semi-supervised-fine-tuning)
  - [4. Configuration Hyperparameter Reference ⚙️](#4-configuration-hyperparameter-reference-️)
    - [4.1 `project`](#41-project)
    - [4.2 `caching`](#42-caching)
    - [4.3 `data`](#43-data)
      - [Recommended data values by profile](#recommended-data-values-by-profile)
    - [4.4 `model`](#44-model)
      - [Recommended model values by profile](#recommended-model-values-by-profile)
    - [4.5 `train`](#45-train)
    - [4.6 `trainer`](#46-trainer)
    - [4.7 `trainer.pseudo_label`](#47-trainerpseudo_label)
  - [5. Recommended Profiles 🎯](#5-recommended-profiles-)
    - [Tiny / Debug](#tiny--debug)
    - [Smoke Test](#smoke-test)
    - [Fast GPU](#fast-gpu)
    - [Full Experiment](#full-experiment)
  - [6. Random Accuracy Debugging Checklist 🔍](#6-random-accuracy-debugging-checklist-)
  - [7. Hyperparameter Sensitivity Summary 📊](#7-hyperparameter-sensitivity-summary-)
  - [8. Known Bug Fixes 🐛](#8-known-bug-fixes-)

---

## 1. System Overview 🔭

MetaGraphSci is a multimodal citation-aware encoder that classifies scientific papers by jointly modeling three complementary signals:

- 📝 **Text** — title and abstract, encoded by a fine-tuned SciBERT backbone with LoRA/QLoRA adapters.
- 🏛️ **Metadata** — venue, publisher, author list, and publication year, encoded by a Deep Cross Network.
- 🕸️ **Citation graph** — the scholarly citation topology, encoded by a hybrid GPS transformer that combines global multi-relational attention with local message passing.

All three streams are fused by a gated residual network and classified by a normalized cosine classifier. Training proceeds in two stages: self-supervised contrastive pretraining using masked-token augmentation and graph-neighborhood positives, followed by semi-supervised fine-tuning with a pseudo-labeling curriculum.

### Data flow

```
documents.csv + citations.csv
         |
         v
  split_documents()          <- time / stratified / random split
         |
         |-- labeled_docs  (label_ratio x train)
         |-- unlabeled_docs
         |-- val_docs
         +-- test_docs
                |
                v
  build_neighbor_cache()      <- local_relevance sampling per split graph
                |
                v
  MultiScaleDocumentDataset   <- tokenized text + metadata + context windows
                |
                v
  MetaGraphSci.forward()
    |-- TextEncoder           -> h_text  [B, text_dim]
    |-- MetadataEncoder       -> h_meta  [B, metadata_dim]
    +-- CitationGraphTransformer -> h_citation [B, citation_dim]
                |
                v
  MultimodalFusion            -> embeddings [B, fusion_dim]
                |
                v
  NormalizedCosineClassifier  -> logits, probabilities
```

---

## 2. Module Reference 🧩

### 2.1 Ablation Contracts

This module defines the shared type and constant contracts used across the entire model.

```python
TensorTriplet: TypeAlias = tuple[Tensor, Tensor, Tensor]  # (h_text, h_meta, h_citation)

ABLATION_MODES: Final[dict[str, frozenset[str]]] = {
    "full":           frozenset({"text", "metadata", "citation"}),
    "text_only":      frozenset({"text"}),
    "text_metadata":  frozenset({"text", "metadata"}),
    "text_citation":  frozenset({"text", "citation"}),
}

NUM_RELATIONS: Final[int] = 4  # citation edges, year proximity, metadata compat, latent adj
```

**Design rationale - Zero-Masked Ablation:** Rather than dynamically rewiring the graph or dropping modules conditionally, the ablation study runs the full forward pass for every modality and then hard-zeros the output tensors for ablated streams. This guarantees consistent batch shapes and pre-trained dimension sizes. The downstream MultimodalFusion Gated Residual Network handles zero-vectors gracefully, perfectly simulating a missing modality.

**`NUM_RELATIONS = 4`** encodes the four distinct relational channels fed to RelationMixer:
1. Structural citation edges (in/out/bidirectional, normalized to `[0, 1]`).
2. Temporal proximity (exponential decay over year delta).
3. Metadata compatibility (cosine similarity between projected venue/publisher/year vectors).
4. Learned latent adjacency (the LatentGraphModule output, active only when `use_latent_graph=True`).

---

### 2.2 TextEncoder 📖

Wraps a SciBERT backbone with optional LoRA, QLoRA, gradient checkpointing, and partial layer freezing.

```python
TextEncoder(
    model_name,                    # Must contain "scibert" — validated at init.
    output_dim,                    # Projects CLS vector to this size (identity if == hidden_size).
    peft_mode,                     # "none" | "lora" | "qlora"
    lora_r, lora_alpha,            # LoRA rank and scaling. Effective scale = lora_alpha / lora_r.
    lora_dropout,
    peft_target_modules,           # Tuple of module names to inject LoRA into. Default: ("query", "value").
    gradient_checkpointing,        # Recompute activations during backward. ~20% slower, large VRAM saving.
    freeze_backbone_until_layer,   # Freeze embedding layer + bottom N encoder blocks.
)
```

**CLS pooling** extracts the representation at position 0:

$$\mathbf{h}_{\text{text}} = \text{SciBERT}(\mathbf{x})[0] \in \mathbb{R}^{d_{\text{text}}}$$

**LoRA** adapts query/value projections with a low-rank decomposition. For a frozen weight matrix $W_0 \in \mathbb{R}^{d \times k}$, the adapted output is:

$$W = W_0 + \frac{\alpha}{r} \cdot BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d,k)$$

**QLoRA path:** When `peft_mode="qlora"`, the backbone is loaded in 4-bit NF4 quantization via `BitsAndBytesConfig`, reducing VRAM by ~75%. `prepare_model_for_kbit_training` is called before `get_peft_model`. Computation remains in bfloat16 (or float16 on older hardware).

**Partial freezing:** Freezes the embedding layer and the bottom `n` transformer blocks. Lower layers encode universal syntactic features; upper layers encode task-specific semantics. Freezing the bottom layers prevents overfitting on small datasets and accelerates training.

> 🐛 **Known bug fixed:** An earlier version assigned the return value of `add_adapter()` to `self.backbone`, replacing the model object with a string and crashing every subsequent call. The current code calls `AutoAdapterModel.from_pretrained()` directly and assigns the model correctly.

---

### 2.3 MetadataEncoder 🏛️

Encodes discrete publication metadata (venue, publisher, authors, year) into a dense vector.

```python
MetadataEncoder(
    num_venues, num_publishers, num_authors,
    embedding_dim,      # Size of each categorical embedding. Year is projected to the same size.
    cross_layers,       # Number of Deep Cross Network layers.
    output_dim,         # Final projection size (= metadata_dim in config).
)
```

**DeepCrossNetwork** models explicit polynomial feature interactions. Each cross layer $l$ computes:

$$\mathbf{x}_{l+1} = \mathbf{x}_0 \left(\mathbf{x}_l^\top \mathbf{w}_l\right) + \mathbf{b}_l + \mathbf{x}_l$$

This produces degree-$(l+1)$ polynomial interactions without exponential parameter growth.

> 🐛 **Known bug fixed:** The original weight initialization used `std=1.0`, causing cross product norms to grow by $\sim16\times$ per layer. After 3 layers this produces a $\sim4096\times$ norm explosion, overflowing to NaN under fp16. The fix initializes with $\sigma_{\text{init}} = 1/\sqrt{d_{\text{in}}}$, keeping each cross product approximately unit-norm.

**Author padding:** `padding_idx=0` is set on the author embedding so empty author slots produce zero vectors. Masked mean pooling then correctly handles variable-length author lists without bias from the padding.

**Fake authors for context metadata:** When encoding context neighbor metadata, author lists are unavailable (they would explode memory). An all-zero author tensor is passed instead. Because `padding_idx=0` causes those entries to produce zero embeddings, and the masked mean ignores zero slots (`author_mask = (author_ids != 0)`), the fake authors mathematically vanish. The venue and publisher signals are still encoded correctly.

---

### 2.4 CitationGraphTransformer 🕸️

The most complex module. Encodes the center document's citation neighborhood into a single fixed-size vector using a learned selection stage, GPS hybrid transformer layers, optional latent graph, and gated pooling.

#### Sub-modules

**SignEquivariantPE** encodes spectral graph features (Laplacian eigenvectors) while being robust to eigenvector sign ambiguity. During training, random $\pm 1$ sign flips are applied to each eigenvector. At inference, absolute values are used. Both strategies produce equivalent MLP inputs up to sign, making the encoding equivariant to the arbitrary sign convention of the eigensolver.

**StructuralPEEncoder** combines four structural signals into one positional representation per context node: edge type embedding, temporal proximity MLP, selector score projection, hop-profile projection (optional), and spectral PE (optional). Outputs are summed and layer-normalized.

> 🐛 **Known bug fixed in StructuralPEEncoder:** When spectral computation is disabled (`spectral_dim=0`), all-zero spectral feature tensors were still passed to the SignEquivariantPE branch. LayerNorm on a near-zero tensor (variance $\to 0$) produces NaN, which propagates through all subsequent attention. The fix is to skip the spectral branch entirely when `spectral_features.abs().max() <= 1e-6`.

**GraphTokenizer** splits each context node into a semantic content token (text projection) and a structural PE token. These are returned separately and injected into the transformer as content + positional bias.

**LearnedCitationSelector** scores all candidate context neighbors and retains the top-$k$ most informative ones. The scoring function fuses: query-candidate semantic similarity (cosine), edge type embedding, temporal proximity, pre-computed cache scores, and a learned MLP. This differentiable selection step is trained end-to-end.

**LatentGraphModule** learns a sparse latent adjacency matrix among the selected context nodes. It computes pairwise attention scores, applies top-$k$ sparsification (retaining only the `latent_graph_top_k` highest scores per node), and produces a soft normalized adjacency. This allows the model to discover same-topic but un-cited paper bridges that do not appear in the observed citation graph.

**RelationMixer** learns per-head mixing weights over the 4 relation channels. For attention head $h$, the relation bias is:

$$\mathbf{B}^{(h)} = \sum_{r=1}^{4} \lambda^{(h)}_r \cdot \mathbf{A}_r, \qquad \boldsymbol{\lambda}^{(h)} = \mathrm{softmax}\!\left(\mathbf{w}^{(h)}_{\text{mix}}\right)$$

Each head can attend to a different weighted combination of citation structure, temporal proximity, metadata compatibility, and latent adjacency.

**LocalMPNNBranch** performs gated local neighborhood aggregation. For each node, aggregates messages from its adjacency-weighted neighbors, then applies a learned gate combining the node's own features with the aggregated message:

$$\mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} a_{ij}\, \mathbf{h}_j, \qquad \mathbf{h}_i' = \sigma(g_i) \odot \mathbf{h}_i + (1 - \sigma(g_i)) \odot \mathbf{m}_i$$

**GPSCitationLayer** is a hybrid relational transformer layer combining:

- Global multi-head attention with relation bias and PE injection:

$$\text{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top + \mathbf{B}^{(h)}_{\text{rel}}}{\sqrt{d_k}}\right)V$$

- Local MPNN aggregation via LocalMPNNBranch.
- Learned gate combining the two outputs:

$$\mathbf{o} = \sigma(g) \cdot \mathbf{o}_{\text{attn}} + (1 - \sigma(g)) \cdot \Delta_{\text{mpnn}}$$

- Feed-forward sublayer with pre-norm.
- PE update: after each layer, node positional encodings are updated by aggregating attention-weighted PE vectors (using detached attention weights to avoid gradient interference).

> 🐛 **Known bug fixed in GPSCitationLayer:** Documents with zero valid context neighbors produce all-$-\infty$ attention logits. $\mathrm{softmax}(-\infty, \ldots, -\infty)$ evaluates to NaN. These NaNs then propagate through the matrix multiplication into the entire batch. The fix is `attn_weights = torch.nan_to_num(attn_weights, nan=0.0)` after the softmax.

**CitationGraphTransformer overall flow:**

```
1. LearnedCitationSelector -> top_idx, top_logits (select top_k from max_context_size candidates)
2. Gather selected embeddings, edge types, year deltas, scores, hop profiles, spectral features
3. GraphTokenizer -> neighbor_feats (content tokens), neighbor_pe (structural PE)
4. center_feat = center_proj(center_text)  # the document being classified
5. Concat [center_feat, neighbor_feats] -> tokens of shape [B, 1 + top_k, output_dim]
6. LatentGraphModule -> latent_adj (if use_latent_graph)
7. build_relation_indicators() -> [B, T, T, 4] relation tensor
8. GPSCitationLayer x num_layers
9. Gated pooling: center_out + gate * mean(neighbor_tokens)
10. output_proj(LayerNorm(pooled))
```

**Hybrid alpha:** The latent adjacency is blended into the relation indicators as:

$$\tilde{\mathbf{A}} = \mathbf{A}_{\text{obs}} + \sigma(\alpha_{\text{hyb}}) \cdot \mathbf{A}_{\text{latent}}$$

where $\alpha_{\text{hyb}}$ is a learnable scalar initialized from `hybrid_alpha_init`. With `hybrid_alpha_init=-2.0`, $\sigma(-2.0) \approx 0.12$, so the model starts heavily weighting observed citations.

---

### 2.5 MultimodalFusion and NormalizedCosineClassifier 🔀

#### MultimodalFusion

```python
MultimodalFusion(
    text_dim, metadata_dim, citation_dim,
    fusion_dim,
    modality_dropout,  # Probability of zeroing an entire modality per sample during training.
)
```

Concatenates the three modality vectors and applies a gated residual network:

$$\mathbf{z} = [\mathbf{h}_{\text{text}},\ \tilde{\mathbf{h}}_{\text{meta}},\ \tilde{\mathbf{h}}_{\text{cit}}]$$

$$\mathbf{e} = \underbrace{W_{\text{res}}\,\mathbf{z}}_{\text{residual}} + \sigma(W_g\,\mathbf{z}) \odot f(W_{\text{mix}}\,\mathbf{z})$$

where $\tilde{\mathbf{h}} = \mathbf{h} \odot \mathbf{m}$, $\mathbf{m} \sim \mathrm{Bernoulli}(1 - p_{\text{drop}})$ (binary mask, training only). Text is never dropped. Randomly zeroing non-text modalities trains the model to degrade gracefully when those streams are absent at inference, matching the behavior of zero-masked ablation.

> 🐛 **Known bug fixed in `maybe_drop`:** The original implementation used `F.dropout(x, p=modality_dropout)`, which scales surviving values by $1/(1-p)$. This meant full-modality batches had $\sim2\times$ the magnitude of modality-dropped batches, systematically miscalibrating the fusion gate. At evaluation (no dropout), the full model scored below `text_only`. The fix is binary masking: `x = x * bernoulli_mask`, so surviving values have exactly the same magnitude as at inference.

#### NormalizedCosineClassifier

```python
NormalizedCosineClassifier(input_dim, num_classes, scale)
```

L2-normalizes both the input embedding and the learnable class prototype vectors, then computes logits as:

$$\text{logits} = s \cdot \hat{\mathbf{e}}\, \hat{W}^\top, \qquad \hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

Logits are bounded to $[-s,\, +s]$. This prevents the classifier from relying on embedding magnitude and makes the loss landscape more uniform across classes. Scale $s$ (typically 8-16) sharpens the softmax to produce meaningful cross-entropy gradients.

---

### 2.6 NeighborhoodAwareContrastiveLoss 📉

`NeighborhoodAwareContrastiveLoss` (NACL) is a modified InfoNCE loss used during contrastive pretraining.

```python
NeighborhoodAwareContrastiveLoss(
    temperature,               # Controls sharpness of similarity distribution.
    metadata_negative_weight,  # Downweight multiplier for metadata-similar negatives.
)
```

**Forward pass logic:**

1. L2-normalize anchor and positive embeddings.
2. Compute all-pairs similarity matrix divided by temperature $\tau$.
3. Subtract row-wise max for numerical stability (the shift cancels in the InfoNCE ratio).
4. Build `negative_mask`: start with all off-diagonal pairs, then remove citation-graph neighbors (preventing false negatives from known-related papers).
5. Apply `metadata_negative_weight` $w_{\text{neg}}$ to pairs sharing venue, publisher, or year — softening rather than removing them, since publication proximity is not a guarantee of semantic similarity.
6. Extract positive scores: diagonal (SimCLR-style anchor vs its augmentation), or from a custom `positive_mask` if graph-adjacent positives are present in the batch.
7. Handle empty positive rows: if `positive_mask` has no valid entries for a row, fall back to the diagonal. Otherwise `(exp * 0).sum() = 0` -> `-log(0) = +inf` -> NaN gradients.
8. Compute weighted InfoNCE:

$$\mathcal{L}_{\text{NACL}} = -\log \frac{\exp\!\left(\mathbf{z}_i \cdot \mathbf{z}_j^+ / \tau\right) + \varepsilon}{\exp\!\left(\mathbf{z}_i \cdot \mathbf{z}_j^+ / \tau\right) + \varepsilon + \displaystyle\sum_{k \notin \mathcal{N}_i} m_k \exp\!\left(\mathbf{z}_i \cdot \mathbf{z}_k / \tau\right)}$$

where $m_k \in \{w_{\text{neg}}, 1\}$ and $\mathcal{N}_i$ is the set of citation-graph neighbors excluded as false negatives.

> 🐛 **Known bug fixed:** The original formula $-\log\bigl(\mathrm{pos} / (\mathrm{pos} + \mathrm{neg} + \varepsilon)\bigr)$ still computes $\log(0)$ when $\mathrm{pos} \approx 0$ (e.g., early training with very low cosine similarity). Adding $\varepsilon$ to the numerator prevents $+\infty$ loss and NaN gradients.

---

### 2.7 PseudoLabeler 🏷️

Generates pseudo-targets for the unlabeled stream during fine-tuning using adaptive confidence thresholding, distribution alignment, and probability sharpening.

```python
PseudoLabeler(
    beta,                   # Threshold multiplier: accept if confidence >= beta * ema_class_max[c].
    warmup_epochs,          # Hard block — no pseudo-labels accepted before this epoch.
    min_per_class,          # Forced minimum per class (0 = disabled, risky if enabled early).
    temperature,            # Sharpening temperature: values < 1.0 sharpen probabilities.
    ema_momentum,           # EMA smoothing for per-class adaptive thresholds.
    distributionalignment,  # Calibrate batch predictions toward the labeled class prior.
    target_prior,           # Expected class distribution (from labeled set).
)
```

**`select()` pipeline:**

```
probs  ->  align()  ->  sharpen()  ->  max(dim=1)  ->  thresholds()  ->  keep mask
```

1. **`align(probs)`** corrects for model bias toward majority classes:

$$\tilde{p}_c = \frac{p_c / \bar{p}_c}{\sum_{c'} p_{c'} / \bar{p}_{c'}}$$

Disabled when `distributionalignment=False`.

2. **`sharpen(probs)`** sharpens the distribution with temperature $T$:

$$\hat{p}_c = \frac{\tilde{p}_c^{1/T}}{\sum_{c'} \tilde{p}_{c'}^{1/T}}$$

Identity at $T=1.0$; sharper peaks at $T < 1.0$.

3. **`thresholds()`** updates the per-class EMA of maximum confidence:

$$\mu_c \leftarrow \eta\,\mu_c + (1-\eta)\max_{\text{batch}} p_c$$

then returns $\beta \cdot \mu_c$ as the per-class acceptance threshold.

4. $\texttt{keep} = \bigl(\max_c \hat{p}_c \;\geq\; \beta \cdot \mu_{\hat{c}}\bigr)$, where $\hat{c} = \arg\max_c \hat{p}_c$.
5. During `epoch <= warmup_epochs`: return an all-False `keep` mask regardless of confidence.
6. If `min_per_class > 0`: for each class with fewer than `min_per_class` accepted samples, forcibly add the top-confidence candidates.

**Persistent adaptive state:** `ema_class_max` must be explicitly saved and restored with the model checkpoint. Losing it silently resets the curriculum to cold-start.

> 🐛 **Known bug fixed:** The original `load_labeler_state_dict` guard was `not isinstance(ema, Tensor)`, so the inner `ema.clone()` branch was entered only when `ema` was **not** a Tensor — always producing `None`. Every checkpoint load silently reset `ema_class_max` to `None`. The fix inverts the condition to `isinstance(ema, Tensor)`.

---

### 2.8 MetaGraphSci Top-Level Model 🏗️

The top-level model class. Owns all three encoders, the fusion layer, and the classifier. Implements the full forward pass and ablation logic.

```python
MetaGraphSci(
    num_classes, num_venues, num_publishers, num_authors,
    text_dim, metadata_dim, citation_dim, fusion_dim,
    classifier_scale,
    model_name, ablation_mode,
    peft_mode, lora_r, lora_alpha, lora_dropout,
    peft_target_modules, gradient_checkpointing, freeze_backbone_until_layer,
    citation_heads, citation_layers, citation_ff_dim,
    selector_hidden_dim, selector_top_k, max_context_size, fusion_modality_dropout,
    metadata_embedding_dim, metadata_cross_layers, citation_dropout,
    hop_profile_dim, spectral_dim, use_latent_graph, latent_graph_top_k,
    num_relations, hybrid_alpha_init,
)
```

**`encode_modalities()`** runs all three encoders and returns their outputs as a `TensorTriplet` after ablation masking:

1. `h_text = text_encoder(input_ids, attention_mask)` — CLS embedding.
2. `h_meta = metadata_encoder(venue_ids, publisher_ids, author_ids, years)` — metadata vector.
3. `candidate_embeddings = citation_encoder.encode_candidates(...)` — batch-encodes all candidate context documents through the shared text encoder.
4. Build `context_meta` from context venue/publisher/year fields using fake all-zero author tensors.
5. `h_citation = citation_encoder(h_text, h_meta, candidate_embeddings, ...)` — graph-aggregated vector.
6. `ablation_study()` — zeros out streams not in the current ablation mode's frozenset.

**`forward()`** calls `encode_modalities()`, then `fusion()`, then `classifier()`. Returns `(embeddings, logits, probabilities)`. With `return_parts=True`, also returns the individual modality vectors for inspection.

**`get_embeddings()`** is a convenience wrapper for contrastive pretraining that returns only the fused embedding.

---

## 3. Training Pipeline 🚀

### 3.1 Data Orchestration

`pipeline.py` is the entry point. It iterates over all `(ablation, seed)` combinations, builds the data bundle for each seed, constructs the model, runs training, evaluates, and saves artifacts.

**`build_run_bundle()`** is the main per-seed data pipeline:

```
load_documents()
    -> assert_label_integrity()    # Labels must be contiguous ints 0..num_classes-1
    -> split_documents()           # time / stratified / random
    -> log_split_diagnostics()     # Log class counts per split before training
    -> create_low_label_split()    # Partition train into labeled + unlabeled
    -> load_or_build_graph()       # Citation graph + split-specific views
    -> create_tokenizer()
    -> load_or_build_encoders()    # venue/publisher/author vocab mappings
    -> load_or_build_tokenization()
    -> load_or_build_doc_embeddings()
    -> build_or_load_cache() x 3  # Per-split neighbor caches
    -> build_dataset() x 5        # pretrain, labeled, unlabeled, val, test
    -> assert labeled split covers all classes (raises early if not)
```

**`assert_label_integrity()`** verifies that all label values form a contiguous integer range `[0, num_classes-1]`. Non-contiguous labels (e.g., class IDs 0, 2, 5) are silently mishandled by `CrossEntropyLoss` — it trains on phantom class slots — producing near-random results with no error message.

**`log_split_diagnostics()`** logs a per-class count table across all three splits before any training begins. This is the primary tool for diagnosing random accuracy from missing classes.

**`labeled_prior()`** computes the empirical class frequency distribution of the labeled subset, passed to PseudoLabeler as `target_prior` for distribution alignment.

**Cache compatibility** is checked by structural fingerprint: `max_context_size`, `sampling_strategy`, `num_nodes`, `k_hops`, `spectral_dim`, `enable_spectral`. If any field mismatches, the cache is rebuilt. Seed is included in neighbor cache metadata but not in the tokenization or embedding caches (which are seed-independent).

**Graph mode:**
- `transductive` — context documents for validation and test splits are drawn from the full document table. Test nodes are visible to the graph at training time (but their labels are withheld).
- `inductive` — each split's context is restricted to its own documents only. Strictly harder; models cannot use test-set structural information.

---

### 3.2 MetaGraphSciTrainerEval ⚙️

Two-stage trainer built on HuggingFace Accelerate. Clean separation between pretraining and fine-tuning.

#### Stage 1 - Contrastive Pretraining

For each batch:

1. Compute anchor embeddings.
2. Augment the batch: randomly mask tokens at rate `ssl_text_dropout` (excluding `[CLS]`, `[SEP]`, and `[PAD]` tokens, which have IDs 101, 102, 0).
3. Compute positive embeddings from the augmented batch.
4. Compute `metadata_affinity` mask: $|y_i - y_j| \leq 2$ (same venue, publisher, or within +-2 years).
5. Build `positive_mask` using graph-adjacency only (not venue/publisher overlap — see bug fix below).
6. Compute NACL loss and backpropagate.

> 🐛 **Known bug fixed in `build_positive_mask`:** The original code required a pair to be both graph-adjacent AND metadata-compatible to qualify as a positive. Their intersection is nearly always empty in a mini-batch, causing the contrastive loss to degenerate to the SimCLR diagonal only — the model never learned from graph topology. The fix uses graph-adjacency alone as the positive criterion. Metadata is still used for negative softening inside NACL.

> 🐛 **Known bug fixed in `metadata_affinity`:** The original year-proximity threshold was `2.0 / 26.0 ~= 0.077`, implying years were normalized over a 26-year range. But raw years (e.g., 2018, 2021) were passed, making the threshold match only papers published within ~2 months. The fix is a direct absolute year delta: $|y_i - y_j| \leq 2.0$.

**Memory optimization:** The pretrain and finetune optimizers are never both alive simultaneously. The pretrain optimizer is deleted and `cuda.empty_cache()` is called before the finetune optimizer is allocated. This halves peak optimizer memory (each AdamW holds $2 \times |\theta|$ moment tensors).

#### Stage 2 - Semi-Supervised Fine-Tuning

For each batch pair (labeled + unlabeled):

1. $\mathcal{L}_{\text{sup}} = \mathrm{CrossEntropy}(\hat{y}_{\text{labeled}},\, y)$ with label smoothing.
2. `unlabeled_logits, unlabeled_probs = forward(unlabeled_batch)`.
3. `pseudo_labeler.select(unlabeled_probs, epoch)` gives `selected` mask and `pseudo_labels`.
4. Optionally filter by `min_pseudo_confidence`.
5. $\mathcal{L}_{\text{ssl}} = \mathrm{CrossEntropy}(\hat{y}_u[\mathcal{S}],\, \tilde{y}[\mathcal{S}])$ (or zero if no samples accepted).
6. $\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda(t) \cdot \mathcal{L}_{\text{ssl}}$

**`pseudo_weight(epoch)`** produces a linear ramp:

$$\lambda(t) = \begin{cases} 0 & t \leq t_{\text{warm}} \\ \lambda_{\text{final}} \cdot \min\!\left(1,\, \dfrac{t - t_{\text{warm}}}{t_{\text{ramp}}}\right) & t > t_{\text{warm}} \end{cases}$$

starting only after `supervised_warmup_epochs` $t_{\text{warm}}$, ramping over `pseudo_ramp_epochs` $t_{\text{ramp}}$.

**Checkpoint selection:** After each epoch, evaluation is called on the validation set. If the `selection_metric` (default `macro_f1`) improves, the model weights and PseudoLabeler state are saved to `best_model.pt`. After fine-tuning, the best checkpoint is reloaded before test evaluation.

**LR schedule:** Linear warmup over `lr_warmup_fraction x total_steps`, then cosine decay to 0. This is the standard BERT fine-tuning recipe. Starting SciBERT at `lr=1e-4` from step 0 corrupts pre-trained representations in the first few gradient steps.

---

## 4. Configuration Hyperparameter Reference ⚙️

### 4.1 `project`

```yaml
project:
  benchmark: "cs_ai"                            # Dataset/experiment profile name. Used in output paths and summaries.
  run_name: "MetaGraphSci_cs_ai_stable"         # Human-readable run identifier for logs and trackers.
  output_dir: "runs/metagraphsci/cs_ai_stable"  # Root for checkpoints, metrics, plots, and artifacts.
  cache_dir: "cache/metagraphsci/cs_ai"         # Root for reusable preprocessing caches.
```

These fields affect paths and logging only — none alter model behavior. `output_dir` must have sufficient disk space; each run writes a full model checkpoint, training history CSV, embedding arrays, confusion matrices, and TSNE/PCA plots. `cache_dir` should persist across runs to avoid re-computing expensive neighbor caches.

---

### 4.2 `caching`

```yaml
caching:
  tokenization_cache: true   # Cache tokenized title+abstract tensors per document.
  doc_embedding_cache: true  # Cache frozen backbone CLS embeddings (seed-independent, invalid during active fine-tuning).
  graph_split_cache: true    # Cache the citation graph object and per-split subgraph views (seed-dependent).
  encoder_cache: true        # Cache venue/publisher/author vocabulary mappings (seed-independent).
  neighbor_cache: true       # Cache pre-ranked context neighbor lists per document (seed-independent for same graph).
```

| Cache | What it stores | Seed-dependent | When to invalidate |
|---|---|---|---|
| `tokenization_cache` | `input_ids`, `attention_mask` tensors | No | tokenizer name, `max_seq_length`, or document table changes |
| `doc_embedding_cache` | Frozen CLS vectors from pretrained backbone | No | model name changes; **must disable during LoRA fine-tuning** |
| `graph_split_cache` | NetworkX graph + split node masks | Yes | `split_strategy`, `test_size`, `val_size` changes |
| `encoder_cache` | Venue/publisher/author string->int mappings | No | document table changes |
| `neighbor_cache` | Pre-ranked top-k neighbors per document | No (for same graph) | sampling weights, `max_context_size`, `k_hops`, `enable_spectral` changes |

⚠️ **Stale cache warning:** The neighbor cache is the most common source of subtle bugs. If `max_context_size` or `sampling_strategy` changes but the cache is not invalidated, the model will train on old neighborhood lists silently. The pipeline does structural fingerprint checking and will log a `MISS` and rebuild when fields change, but only for cached fields — set all caches to `false` when doing early debugging to be safe.

---

### 4.3 `data`

```yaml
data:
  documents: "data/cs_ai/documents.csv"  # Normalized document table (one row per paper).
  citations:  "data/cs_ai/citations.csv" # Citation edge table (source_col -> target_col).
  baselines:  ""                         # Optional path to pre-computed baselines for comparison tables.

  label_column: "label"    # Integer class target column in documents.csv. Must be contiguous 0..C-1.
  source_col:   "source"   # Citing document ID column in citations.csv.
  target_col:   "target"   # Cited document ID column in citations.csv.

  split_strategy: "time"         # Partition strategy: random | stratified | time.
  graph_mode:     "transductive" # Graph visibility: transductive | inductive.

  label_ratio: 0.25  # Fraction of the training set with labels. Critical for avoiding random accuracy.
  test_size:   0.20  # Fraction of all documents held out for final test evaluation.
  val_size:    0.10  # Fraction of training documents held out for validation.

  max_seq_length: 256  # Maximum token length for title + abstract.
  max_authors:   null  # Maximum author IDs per document (null = infer from 95th percentile, capped at 32).

  max_context_size: 8          # Maximum citation neighbors per document. Controls memory and speed.
  k_hops:           2          # Depth of hop-profile features per neighbor.
  spectral_dim:     0          # Laplacian eigenvector features per node (0 = disabled).

  sampling_strategy:    "local_relevance" # Neighbor ranking: local_relevance | uniform | random.
  connectivity_weight:   0.45             # Score weight for node degree / connectivity.
  temporal_weight:       0.25             # Score weight for publication-year proximity.
  reciprocity_weight:    0.10             # Score weight for bidirectional (mutual) citation.
  overlap_weight:        0.20             # Score weight for shared-neighborhood overlap (co-citation).

  enable_spectral:            false  # Compute Laplacian eigenvectors for spectral PE.
  hub_degree_threshold:       0      # Exclude nodes above this degree (0 = disabled).
  max_graph_nodes_for_hops:   5000   # Hard cap on graph size for hop-profile computation.
  cache_n_jobs:              -1      # CPU workers for neighbor cache construction (-1 = all cores).

  cache_text:          true   # Keep tokenized tensors in dataset memory.
  pretokenize_context: true   # Tokenize all context documents at dataset construction time.
```

**`label_ratio`** is the single most impactful setting for avoiding random accuracy. At 5% with 40 classes, the labeled split likely has fewer than 5 examples per class. The pipeline will raise a `ValueError` if any class is entirely absent from the labeled set.

**`split_strategy`:**
- `time` — Chronological split. Simulates realistic deployment where the model is evaluated on future papers. Requires a year column.
- `stratified` — Class-proportional random split. Use when temporal ordering is not relevant.
- `random` — Fully random. Only for ablations without stratification.

**`graph_mode`:**
- `transductive` — Test node embeddings (without labels) are visible to the graph encoder during training, providing structural context. Standard for citation benchmarks.
- `inductive` — Test nodes are completely hidden. Strictly harder; models cannot use test-set graph topology.

**Sampling weights** control the `local_relevance` scoring function. They do not need to sum to exactly 1.0 — the scorer normalizes internally. Increasing `overlap_weight` favors co-cited papers (strong topical signal). Increasing `temporal_weight` biases toward same-era papers.

**`enable_spectral` / `spectral_dim`:** Disabled by default. Spectral computation on large graphs is expensive and the gains are marginal. Enable only after the base model is verified working.

**`hub_degree_threshold`:** Prevents high-degree survey or review papers from dominating every document's context window. Keep at 0 unless the dataset has extreme hub nodes skewing neighbor quality.

#### Recommended data values by profile

| Field | Tiny/Debug | Smoke | Fast GPU | Full Experiment |
|---|---|---|---|---|
| `label_ratio` | 0.30 | 0.25 | 0.20 | 0.20 |
| `max_seq_length` | 128 | 192 | 256 | 320 |
| `max_context_size` | 2 | 4 | 8 | 12 |
| `k_hops` | 1 | 2 | 2 | 2 |
| `pretokenize_context` | false | true | true | true |

---

### 4.4 `model`

```yaml
model:
  tokenizer_name: "allenai/scibert_scivocab_uncased"  # HuggingFace backbone and tokenizer identifier.

  text_dim:     768   # Output dimension of TextEncoder. Must match SciBERT hidden size (768) unless a projection is added.
  metadata_dim: 256   # Output dimension of MetadataEncoder.
  citation_dim: 256   # Output dimension of CitationGraphTransformer.
  fusion_dim:   512   # Input size of the classifier and output of MultimodalFusion.

  metadata_embedding_dim: 64  # Categorical embedding size for venue, publisher, author, year.
  metadata_cross_layers:   2  # Number of Deep Cross Network layers.

  classifier_scale: 8.0  # Logit scale multiplier for NormalizedCosineClassifier. See sensitivity table.

  peft_mode:                  "lora"   # PEFT: none | lora | qlora.
  lora_r:                      8       # LoRA rank. Effective scale = lora_alpha / lora_r = 2.0.
  lora_alpha:                 16       # LoRA scaling factor.
  lora_dropout:                0.1     # Dropout inside LoRA adapters.
  gradient_checkpointing:      true    # Recomputes activations during backward to save VRAM (~20% slower).
  freeze_backbone_until_layer: 6       # Freeze embedding + bottom N transformer blocks.

  citation_heads:    4    # Attention heads in CitationGraphTransformer layers.
  citation_layers:   2    # Number of GPSCitationLayer stacks.
  citation_ff_dim:   512  # Feed-forward size in GPSCitationLayer.
  citation_dropout:  0.15 # Dropout in citation encoder.

  selector_hidden_dim: 256  # Hidden size for LearnedCitationSelector MLP.
  selector_top_k:        6  # Number of neighbors passed to the transformer (must be <= max_context_size).

  fusion_modality_dropout: 0.15  # Probability of zeroing metadata or citation stream per sample during training.

  use_latent_graph:    true   # Enable LatentGraphModule for learned adjacency.
  latent_graph_top_k:     3   # Latent neighbors per node.
  hybrid_alpha_init:  -2.0   # Initial log-sigmoid weight: sigmoid(-2.0) ~= 0.12 (favors observed citations).
```

**`classifier_scale`:** The most sensitive architectural hyperparameter for getting off random accuracy. Logits are bounded to $[-s, +s]$. Too high ($s > 20$): near-one-hot softmax, vanishing CE gradients for non-peak classes. Too low ($s < 2$): near-uniform softmax, extremely slow learning. Start at 8.0; reduce to 4.0 if stuck near random; increase to 12-16 only after the model is clearly learning.

**LoRA:** `lora_r=8`, `lora_alpha=16` gives an effective scale of $\alpha/r = 2.0$ — conservative and well-tested. Do not increase `lora_r` beyond 16 without evidence the task requires higher capacity; larger rank negates the VRAM savings.

**`freeze_backbone_until_layer`:** Freeze 6 of 12 SciBERT layers when fine-tuning on small datasets (< 5,000 labeled examples). Set to 0 to allow all layers to adapt via LoRA. Set to 11 or 12 to train only the classification head (useful for initial debugging).

**`use_latent_graph`:** Disable when debugging to reduce forward-pass cost. The latent graph is the most expensive citation encoder component. It can be enabled after the rest of the model is verified working.

**`hybrid_alpha_init`:** Passed through a sigmoid. Set to 0.0 for equal weighting of observed and latent graphs at initialization. Positive values favor the latent graph; negative values favor observed citations.

#### Recommended model values by profile

| Field | Text-Only Debug | Low-Memory GPU | Full Multimodal |
|---|---|---|---|
| `citation_layers` | 0 (via text_only ablation) | 1 | 2 |
| `citation_heads` | — | 2 | 4 |
| `freeze_backbone_until_layer` | 11 | 8 | 6 |
| `use_latent_graph` | false | false | true |
| `gradient_checkpointing` | false | true | true |
| `peft_mode` | none | lora | lora |
| `fusion_dim` | 256 | 256 | 512 |

---

### 4.5 `train`

```yaml
train:
  batch_size:      8   # Per-device batch size.
  num_workers:     2   # DataLoader worker processes.
  pretrain_epochs: 3   # Contrastive pretraining epochs before fine-tuning.
  finetune_epochs: 25  # Supervised fine-tuning epochs.
  seeds:
    - 42
    - 1337
    - 2025
  ablations:
    - text_only
    - text_metadata
    - full
```

**`ablations`:** Always run `text_only` before `full`. A well-tuned text-only model provides a reliable lower bound and sanity check. If `full` scores below `text_only`, the graph/metadata encoders are hurting — diagnose dimension mismatches, `classifier_scale`, or stale caches before continuing.

**`pretrain_epochs`:** Set to 0 to skip pretraining entirely for fast debugging. 3-5 epochs is sufficient for most benchmarks; the contrastive stage warms up representations but is not the primary training signal.

**Effective batch size:** `batch_size x gradient_accumulation_steps`. With `batch_size=8` and `gradient_accumulation_steps=4`, the effective batch is 32. Reduce `batch_size` for OOM; increase `gradient_accumulation_steps` to compensate.

**Multi-seed experiments:** Three seeds is the minimum for reliable benchmarking. Single-seed results should be labeled preliminary. The pipeline automatically handles multi-seed execution, checkpointing, and metric aggregation.

---

### 4.6 `trainer`

```yaml
trainer:
  mixed_precision:              "bf16"  # Precision: no | fp16 | bf16.
  gradient_accumulation_steps:   4      # Steps before optimizer update. Effective batch = batch_size x steps.
  pretrain_lr:                   1.0e-5 # Learning rate for contrastive pretraining.
  finetune_lr:                   2.0e-5 # Learning rate for supervised fine-tuning.
  weight_decay:                  0.01   # AdamW L2 regularization.
  max_grad_norm:                 1.0    # Gradient clipping norm.
  lr_warmup_fraction:            0.08   # Fraction of total steps for linear LR warmup.
  selection_metric:              "macro_f1"  # Validation metric for best-checkpoint selection.
  label_smoothing:               0.05        # CrossEntropy label smoothing.

  lambda_ssl:                    0.0    # Initial pseudo-label loss weight.
  lambda_ssl_final:              0.25   # Final pseudo-label loss weight after ramp.
  supervised_warmup_epochs:      5      # Supervised-only epochs before SSL ramp begins.
  pseudo_ramp_epochs:            8      # Epochs to ramp lambda from 0 to lambda_ssl_final.
  min_pseudo_confidence:         0.7    # Minimum predicted max-probability to accept a pseudo-label.

  contrastive_temperature:       0.1    # Temperature for NACL similarity logits.
  metadata_negative_weight:      0.25   # Downweight for metadata-similar negatives in NACL.
  ssl_text_dropout:              0.1    # Token dropout rate for contrastive augmentation.

  use_mlflow:  true
  mlflow_experiment: "MetaGraphSci_openalex_ai"
  use_wandb:   false
```

**Learning rates:** `pretrain_lr` should be at most `3e-5`; higher values cause representation collapse during contrastive pretraining. `finetune_lr` in the range `1e-5`-`3e-5` is standard for LoRA fine-tuning; above `5e-5` risks catastrophic forgetting of the SciBERT vocabulary.

**`selection_metric: macro_f1`:** Use macro F1 rather than accuracy for imbalanced classes. Accuracy is dominated by majority classes and can report high values even when minority classes are predicted at chance.

**Mixed precision:**
- `no` — Full float32. Use on CPU or when debugging NaN gradients.
- `fp16` — For Volta/Turing GPUs. The $1/\sqrt{d}$ DCN initialization prevents fp16 overflow.
- `bf16` — Preferred for Ampere+ GPUs. Larger dynamic range than fp16; use whenever available.

**Pseudo-labeling schedule** (`lambda_ssl`, `lambda_ssl_final`, `supervised_warmup_epochs`, `pseudo_ramp_epochs`):

```
Epoch:  1 ---- supervised_warmup_epochs ---- +pseudo_ramp_epochs ---- ...
Weight: 0                  0              ->   lambda_ssl_final        lambda_ssl_final
```

Set `lambda_ssl: 0.0` and `lambda_ssl_final: 0.0` to disable pseudo-labeling completely during debugging. Enable only after the supervised model achieves clearly above-random macro F1.

**`contrastive_temperature`:** Lower values (0.05-0.10) make the loss more selective. Below 0.05 risks NaN loss early in training. Start at 0.10.

---

### 4.7 `trainer.pseudo_label`

```yaml
trainer:
  pseudo_label:
    beta:                  0.9    # Acceptance threshold multiplier: accept if confidence >= beta * ema_class_max[c].
    warmup_epochs:         5      # Hard block — no pseudo-labels before this epoch, regardless of confidence.
    min_per_class:         0      # Forced minimum pseudo-labels per class (0 = disabled).
    temperature:           0.75   # Sharpening temperature. Values < 1.0 produce sharper, more decisive probabilities.
    ema_momentum:          0.98   # EMA smoothing for per-class adaptive thresholds.
    distributionalignment: false  # Calibrate predictions toward labeled class prior.
```

**`beta`:** Conservative values (0.90-0.95) accept fewer but more reliable pseudo-labels. Aggressive values (0.70-0.80) accept more labels earlier, useful when the labeled set is very small. For unstable classifiers, 0.80-0.85 is safer.

**`warmup_epochs`:** A hard block that overrides all confidence thresholds. Always set `>= trainer.supervised_warmup_epochs`. Accepting pseudo-labels before the classifier is stable causes confirmation collapse: the model reinforces its own early errors.

**`min_per_class`:** When greater than 0, forcibly adds the top-confidence candidates per class to meet the minimum. This is risky — if the model is wrong about a class, it inserts bad labels. Keep at 0 until the model is reliably above random.

**`temperature`:** 0.5 produces sharply peaked probability distributions ($\hat{p}_c \propto p_c^{2}$). 1.0 is the identity. Sharper distributions are more reliable as pseudo-labels but more susceptible to early overconfidence.

**`distributionalignment`:** Corrects for model bias toward majority classes by calibrating the batch distribution toward the labeled prior. Disable when the labeled class distribution is unreliable or highly imbalanced — alignment toward a bad prior can degrade predictions rather than improve them.

**`ema_momentum`:** Controls adaptation speed $\eta$ of per-class thresholds. High values (0.95-0.98) produce stable, slow-adapting thresholds. Low values (0.80) adapt quickly but can chase noise.

---

## 5. Recommended Profiles 🎯

### Tiny / Debug

Goal: prove the text classifier can learn at all. If it cannot, all other experiments are unreliable.

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
  lambda_ssl: 0.0
  lambda_ssl_final: 0.0
  mixed_precision: "no"

model:
  freeze_backbone_until_layer: 11
  use_latent_graph: false
```

✅ Expected: loss decreases, macro F1 clearly above $1/C$ within 3 epochs. If not, fix the data pipeline (label encoding, class coverage) before touching the model.

### Smoke Test

Goal: verify the full pipeline runs end-to-end without errors.

```yaml
data:
  label_ratio: 0.25
  max_context_size: 4
  max_seq_length: 192

train:
  ablations: ["text_only", "full"]
  batch_size: 8
  pretrain_epochs: 1
  finetune_epochs: 5

trainer:
  lambda_ssl: 0.0
  lambda_ssl_final: 0.0
```

✅ Expected: both ablations complete without errors. `full` macro F1 >= `text_only`. If `full` is worse, diagnose the citation encoder.

### Fast GPU

Goal: practical iteration with full features.

```yaml
data:
  label_ratio: 0.20
  max_context_size: 8
  max_seq_length: 256

train:
  ablations: ["text_only", "full"]
  batch_size: 16
  pretrain_epochs: 2
  finetune_epochs: 15

trainer:
  mixed_precision: "bf16"
  lambda_ssl_final: 0.15
```

✅ Expected: full model clearly outperforms text-only. Reasonable macro F1 within ~2 hours on a single A100. Use this profile for most hyperparameter search and ablation work.

### Full Experiment

Goal: final benchmark reporting with multi-seed reproducibility.

```yaml
data:
  label_ratio: 0.20
  max_context_size: 12
  max_seq_length: 320

train:
  seeds: [42, 1337, 2025]
  ablations: ["text_only", "text_metadata", "full"]
  batch_size: 16
  pretrain_epochs: 5
  finetune_epochs: 30

trainer:
  lambda_ssl_final: 0.25
  supervised_warmup_epochs: 5
  pseudo_ramp_epochs: 8
  mixed_precision: "bf16"
```

✅ Expected: three-seed mean +- std reported for all ablations. Pseudo-labeling ramped in after warmup. Run only after smoke and fast GPU profiles have passed cleanly.

---

## 6. Random Accuracy Debugging Checklist 🔍

If validation macro F1 is near $1/C$, follow these steps in order:

**1️⃣ Run `text_only` first.** Isolates the text classifier from graph and metadata. If `text_only` is also near random, the problem is in data loading, label encoding, or the text encoder — not the graph.

**2️⃣ Disable pseudo-labeling completely.** Set `lambda_ssl: 0.0` and `lambda_ssl_final: 0.0`. Pseudo-labels from a random classifier amplify noise and prevent learning.

**3️⃣ Increase `label_ratio` to 0.25 or 0.30.** At low ratios, some classes may have zero labeled examples. The pipeline raises an error for this, but borderline coverage (1-2 examples per class) can still produce near-random results.

**4️⃣ Verify labels are contiguous integers from 0 to `num_classes - 1`.** Non-contiguous labels are silently mishandled by `CrossEntropyLoss`. Use `assert_label_integrity()` output in logs to confirm.

**5️⃣ Verify every class appears in the labeled training split.** Run `log_split_diagnostics()` output (logged at the start of every run) and check the `train` column for zeros.

**6️⃣ Overfit 32 examples.** Fix a mini-batch of 32 labeled examples and run for 50 epochs with `lr_warmup_fraction: 0.0`. The model must achieve near-100% training accuracy. If it does not, the architecture or loss function has a fundamental error.

**7️⃣ Reduce `classifier_scale`.** Values above 16 cause near-one-hot softmax, collapsing gradient signal. Try `classifier_scale: 4.0`.

**8️⃣ Reduce `max_context_size` to 2.** Large context windows introduce noisy neighbors before attention weights are calibrated.

**9️⃣ Disable graph/citation modules.** Set `ablation_mode: text_only`. If text-only works and full does not, the citation encoder is the source of the problem.

**🔟 Check macro F1, not only accuracy.** High accuracy on imbalanced datasets can coexist with near-random per-class recall on minority classes.

---

## 7. Hyperparameter Sensitivity Summary 📊

| Hyperparameter | Too Low | Too High | Safe Starting Value |
|---|---|---|---|
| `label_ratio` | No examples per class; random accuracy | Memory pressure | 0.20-0.25 |
| `max_seq_length` | Truncates key abstract content | OOM; slow tokenization | 256 |
| `max_context_size` | Insufficient graph signal | OOM; noisy neighbors | 8 |
| `classifier_scale` | Uniform softmax; slow learning | Near-one-hot; vanishing gradients | 8.0 |
| `pretrain_lr` | Slow contrastive convergence | Representation collapse | 1e-5 |
| `finetune_lr` | Slow convergence | Catastrophic forgetting | 2e-5 |
| `lambda_ssl_final` | No pseudo-label benefit | Noisy labels overpower supervised signal | 0.15-0.25 |
| `supervised_warmup_epochs` | Pseudo-labels from unstable model; collapse | Wasted training time | 5 |
| `pseudo_ramp_epochs` | Abrupt SSL introduction; instability | Delayed benefit | 6-10 |
| `min_pseudo_confidence` | Low-quality labels accepted | Very few labels accepted | 0.70-0.80 |
| `contrastive_temperature` | NaN loss; instability | Loose separation; weak signal | 0.10 |
| `label_smoothing` | Overconfident predictions | Blurred class boundaries | 0.05 |
| `batch_size` | High gradient variance | OOM | 8-16 |
| `lora_r` | Underfitting for very task-specific vocabulary | Negates VRAM savings | 8 |
| `metadata_cross_layers` | Underfits feature interactions | fp16 norm explosion risk | 2-3 |
| `citation_layers` | Shallow graph reasoning | Oversmoothing; slow training | 2 |
| `fusion_modality_dropout` | Overreliance on all modalities | Underfitting multimodal signal | 0.10-0.20 |

---

## 8. Known Bug Fixes 🐛

This section consolidates all bugs identified and fixed in the codebase, as documented in source comments.

| Module | Bug | Fix |
|---|---|---|
| TextEncoder | `add_adapter()` return value (adapter name string) was assigned to `self.backbone`, replacing the model object. | Call `AutoAdapterModel.from_pretrained()` directly; never reassign `self.backbone` from adapter calls. |
| MetadataEncoder | DCN weight init with `std=1.0` caused ~4096x norm explosion after 3 layers, overflowing fp16 to NaN. | Initialize with $\sigma = 1/\sqrt{d_{\text{in}}}$. |
| MultimodalFusion | `F.dropout` on modality vectors scaled surviving values by $1/(1-p)$, causing 2x magnitude mismatch between train/eval. Full model scored below `text_only` at evaluation. | Binary masking: `x = x * bernoulli_mask`. No scaling applied. |
| CitationGraphTransformer (StructuralPEEncoder) | All-zero spectral features caused LayerNorm variance $\to 0$ $\to$ NaN, propagating through all attention. | Skip spectral branch when `spectral_features.abs().max() <= 1e-6`. |
| CitationGraphTransformer (GPSCitationLayer) | Documents with zero valid neighbors produce all-$-\infty$ attention logits. $\mathrm{softmax}(-\infty) = \mathrm{NaN}$, corrupting entire batch. | `attn_weights = torch.nan_to_num(attn_weights, nan=0.0)` after softmax. |
| NeighborhoodAwareContrastiveLoss | $-\log\bigl(\mathrm{pos} / (\mathrm{pos} + \mathrm{neg} + \varepsilon)\bigr)$ still computes $\log(0)$ when $\mathrm{pos} \approx 0$ early in training $\to +\infty$ loss $\to$ NaN gradients. | Add $\varepsilon$ to numerator: $-\log\bigl((\mathrm{pos}+\varepsilon) / (\mathrm{pos} + \mathrm{neg} + \varepsilon)\bigr)$. |
| PseudoLabeler | `load_labeler_state_dict` guard was `not isinstance(ema, Tensor)`, always resetting `ema_class_max` to `None` on checkpoint load. | Invert condition to `isinstance(ema, Tensor)`. |
| TrainerEval (metadata_affinity) | Year proximity threshold $2.0/26.0 \approx 0.077$ assumed normalized years; raw years were passed, making the window ~2 months instead of 2 years. | Direct absolute delta: $|y_i - y_j| \leq 2.0$. |
| TrainerEval (build_positive_mask) | Required BOTH graph-adjacency AND metadata compatibility as the positive criterion. Intersection nearly always empty in a mini-batch -> contrastive loss degenerated to SimCLR diagonal only. | Graph-adjacency alone as positive criterion. Metadata used only for negative softening in NACL. |
| TrainerEval (train_full_pipeline) | Both AdamW optimizers were created simultaneously before `accelerator.prepare()`, doubling peak optimizer memory. Fused AdamW also received CPU parameters. | Prepare model and loaders first. Create and prepare each optimizer just before its own stage; delete previous optimizer before allocating next. |
