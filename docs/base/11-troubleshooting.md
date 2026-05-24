# Troubleshooting

[Previous](./10-configuration.md) · [Index](./00-index.md) · [Next](./12-roadmap.md)

---

## :wrench: Debugging Modules

This page covers issues that match the uploaded MetaGraphSci code.

## SciBERT model loading fails

Check:

- `model_name` contains `scibert`,
- the required Hugging Face/adapters packages are installed,
- the model is available locally or can be downloaded,
- `peft_mode` is one of the modes supported by the training script.

## QLoRA or low-bit loading fails

Check:

- CUDA availability,
- bitsandbytes installation,
- GPU support for the selected compute dtype,
- whether `peft_mode="qlora"` is intended for the current run.

## Shape mismatch in fusion

Check that the encoder output dimensions match:

```text
text_dim
metadata_dim
citation_dim
fusion_dim
```

Also verify that the fusion input is exactly:

```text
[text representation, metadata representation, citation representation]
```

## Invalid ablation mode

Valid modes are:

```text
full
text_only
text_metadata
text_citation
```

Any unknown mode should be treated carefully because it may silently fall back to the full mode depending on how the local code is called.

## NaN in graph encoder

Check:

- all-masked context rows,
- empty spectral features,
- invalid adjacency masks,
- attention logits containing only `-inf`,
- citation context masks with no valid neighbors.

## NaN in metadata encoder

Check:

- very large metadata feature magnitudes,
- unstable cross-network initialization,
- invalid year values,
- empty author lists not using the expected zero padding.

## Pseudo-labeling accepts too few samples

Check:

- warmup epoch setting,
- `beta`,
- temperature,
- target prior,
- class imbalance,
- whether `ema_class_max` was restored from checkpoint.

## Evaluation files are missing

Check whether the evaluation function received:

- `y_true`,
- `y_pred`,
- optional `y_prob`,
- optional embeddings,
- optional history rows.

Calibration plots require probabilities. Embedding plots require stored embeddings. Training-history plots require history rows.
