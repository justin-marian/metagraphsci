# Cache layer: tokenization, embeddings, encoders, graph, neighbours

[Previous](./04-datasets.md) · [Index](./00-index.md) · [Next](./06-experiments.md)

---

## Cache layer

The cache layer is the main correctness and speed mechanism. Each artifact stores tensors plus compatibility metadata. If the current run signature matches the saved sidecar, the cache is reused.

| Cache | File | Invalidated by | Incremental |
|---|---|---|---|
| Tokenization | `tokens.pt` | tokenizer, sequence length, document text | yes |
| Embeddings | `embeddings.pt` | model, pooling, sequence length, document text | yes |
| Encoders | `encoders.json` | train split, metadata fields, seed | no |
| Graph | `graph.pt` | citation edges, documents, split settings | no |
| Neighbours | `neighbors.json` | graph, years, scoring weights, context size | no |

## Compatibility check

A cache is valid only when:

$$
\operatorname{meta}_{saved}[k] = \operatorname{meta}_{expected}[k] \quad \forall k \in \mathcal{K}_{compat}
$$

## Neighbour scoring

The neighbour cache ranks citation context with:

$$
s(i,j)=\lambda_d d(j)+\lambda_y e^{-|y_i-y_j|/\tau}+\lambda_r r(i,j)+\lambda_o J(\mathcal{N}_i,\mathcal{N}_j)
$$

where \(d(j)\) is normalized degree, \(e^{-|y_i-y_j|/\tau}\) favours close publication years, \(r(i,j)\) captures reciprocity, and \(J\) is one-hop Jaccard overlap.

## Practical rule

Never edit `*.meta.json` sidecars to force compatibility. Rebuild the cache instead.

---

[Previous](./04-datasets.md) · [Index](./00-index.md) · [Next](./06-experiments.md)
