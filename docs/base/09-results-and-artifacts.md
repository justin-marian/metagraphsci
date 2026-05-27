# Results and artifacts

[Previous](./08-public-module-api.md) · [Index](./00-index.md) · [Next](./10-configuration.md)

---

## Core evaluation artifacts

For each evaluated split, save:

```text
{split}_metrics.json
{split}_per_class.csv
{split}_predictions.csv
{split}_confusion.png
{split}_class_support.png
{split}_per_class_f1.png
{split}_confidence.png
```

Optional outputs include calibration plots, UMAP or t-SNE projections, training curves, and pseudo-label ratio plots.

<p align="center">
  <img src="../../images/EmbeddingClasses.jpg" alt="Embedding projection artifact" width="95%">
</p>

## Comparison rule

Do not compare runs as the same experiment if any of these changed: ablation mode, text backbone, PEFT mode, citation context size, latent graph setting, metadata vocabulary, label mapping, split, pseudo-label thresholds, or random seed.

## Metric definitions

Macro-F1 averages class-wise F1 equally:

$$
F1_{macro}=\frac{1}{C}\sum_{c=1}^{C}\frac{2P_cR_c}{P_c+R_c}
$$

Balanced accuracy averages class-wise recall:

$$
BA=\frac{1}{C}\sum_{c=1}^{C}R_c
$$

---

[Previous](./08-public-module-api.md) · [Index](./00-index.md) · [Next](./10-configuration.md)
