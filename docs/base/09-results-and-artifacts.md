# Results and artifacts

[Previous](./08-public-module-api.md) · [Index](./00-index.md) · [Next](./10-configuration.md)

---

## :bar_chart: Result tables & images

MetaGraphSci result files should reflect the available evaluation utilities.

## Core evaluation artifacts

For each evaluated split, the evaluation bundle can save:

```text
{split}_metrics.json
{split}_per_class.csv
{split}_predictions.csv
{split}_confusion.png
{split}_class_support.png
{split}_per_class_f1.png
{split}_confidence.png
```

## Optional artifacts

When the required inputs are available, additional outputs can be saved:

```text
{split}_calibration.png
{split}_umap.png
training_curves.png
pseudo_label_ratio.png
```

## Recommended run context

Save these fields next to every experiment result:

```text
dataset=<dataset name>
split=<train/validation/test>
model=MetaGraphSci
ablation_mode=<full/text_only/text_metadata/text_citation>
text_encoder=<SciBERT model name>
peft_mode=<none/lora/qlora>
fusion_dim=<value>
classifier_scale=<value>
citation_layers=<value>
citation_heads=<value>
selector_top_k=<value>
max_context_size=<value>
metadata_embedding_dim=<value>
metadata_cross_layers=<value>
use_latent_graph=<true/false>
latent_graph_top_k=<value>
loss=<supervised/contrastive/semi-supervised combination>
seed=<value>
```

## Comparison rule

Do not compare results as the same experiment if any of these changed:

- ablation mode,
- text backbone,
- PEFT mode,
- citation context size,
- latent graph setting,
- metadata vocabulary mapping,
- label mapping,
- train/validation/test split,
- pseudo-labeling thresholds,
- random seed.
