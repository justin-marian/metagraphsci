# Experiments and evaluation flow

[Previous](./05-cache-layer.md) · [Index](./00-index.md) · [Next](./07-model-inspection.md)

---

## Experiment goals

MetaGraphSci experiments should measure classification quality, ablation behaviour, representation quality, pseudo-label stability, and numerical robustness.

<p align="center">
  <img src="../../images/losses.png" alt="Training objective overview" width="95%">
</p>

## Ablation order

| Mode | Text | Metadata | Citation | Question answered |
|---|---:|---:|---:|---|
| `text_only` | yes | no | no | How strong is SciBERT alone? |
| `text_metadata` | yes | yes | no | Does bibliographic metadata help? |
| `text_citation` | yes | no | yes | Does graph context help? |
| `full` | yes | yes | yes | How strong is the complete model? |

## Loss composition

A complete run can combine three terms:

$$
\mathcal{L}=\mathcal{L}_{sup}+\lambda_{ssl}\mathcal{L}_{graph}+\lambda_{pl}\mathcal{L}_{pseudo}
$$

## Evaluation bundle

Save aggregate metrics, per-class metrics, predictions, confusion matrix, class support, per-class F1, confidence histogram, optional calibration, optional embedding projection, and training-history plots.

---

[Previous](./05-cache-layer.md) · [Index](./00-index.md) · [Next](./07-model-inspection.md)
