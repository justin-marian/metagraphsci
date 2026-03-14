# Datasets for scientific citation recommendation

This section summarizes the main datasets and evaluation resources that are suitable for experiments in scientific citation recommendation, paper retrieval, and document representation learning with SciBERT.

The resources below serve two different purposes:

- **Training corpora**, used to construct citation pairs, candidate sets, and retrieval experiments
- **Evaluation benchmarks**, used to measure the quality of learned scientific document embeddings

## Overview

| Resource | Official link | Category | Recommended use | Notes |
|---|---|---|---|---|
| **S2ORC** | [AllenAI S2ORC](https://github.com/allenai/s2orc) | Training corpus | Primary large-scale training corpus | Suitable for building citation-based supervision from scientific metadata and citation structure |
| **OpenAlex** | [OpenAlex Docs](https://docs.openalex.org/) | Training corpus / metadata source | Open and reproducible dataset construction | Good choice for creating train, validation, and test splits from scholarly metadata and citation links |
| **OpenCitations** | [OpenCitations Downloads](https://download.opencitations.net/) | Citation graph | Citation-link supervision and graph-based experiments | Useful when citation edges are the main signal and metadata is obtained from another source |
| **ACL Anthology** | [ACL Anthology Data Access](https://aclanthology.org/faq/data/) | Domain-specific corpus | Small-scale pilot experiments in NLP | Suitable for fast prototyping in a clean and focused research domain |
| **SciRepEval** | [AllenAI SciRepEval](https://github.com/allenai/scirepeval) | Evaluation benchmark | Main benchmark for representation learning evaluation | Covers retrieval, search, classification, and regression tasks for scientific documents |
| **SciDocs** | [AllenAI SciDocs](https://github.com/allenai/scidocs) | Evaluation benchmark | Secondary benchmark for comparison with prior work | Useful for historical comparison with older scientific embedding approaches |
| **PeerRead** | [AllenAI PeerRead](https://github.com/allenai/PeerRead) | Review dataset | Optional extension dataset | More relevant for peer-review and acceptance-related experiments than for citation ranking alone |

## Recommended experimental usage

### Training resources

| Goal | Recommended resource | Alternative |
|---|---|---|
| Large-scale citation recommendation training | **S2ORC** | **OpenAlex** |
| Fully open and reproducible dataset pipeline | **OpenAlex** | **OpenCitations** combined with an external metadata source |
| Small pilot experiment in a focused domain | **ACL Anthology** | **OpenAlex subset** |

### Evaluation resources

| Goal | Recommended resource | Alternative |
|---|---|---|
| Main evaluation of scientific document embeddings | **SciRepEval** | **SciDocs** |
| Comparison with older representation learning work | **SciDocs** | — |
| Review-aware or decision-related experiments | **PeerRead** | — |

## Resource selection guide

| Use case | Recommended resource |
|---|---|
| Best large-scale training corpus | **S2ORC** |
| Best open scholarly metadata source | **OpenAlex** |
| Best open citation graph | **OpenCitations** |
| Best small domain-specific corpus for NLP | **ACL Anthology** |
| Best benchmark for embedding evaluation | **SciRepEval** |
| Best legacy benchmark for comparison | **SciDocs** |
| Best optional dataset for review-related extensions | **PeerRead** |

## Suggested setup for this project

For a SciBERT-based citation recommendation pipeline, a practical setup is:

- **S2ORC** or **OpenAlex** to build positive citation pairs and candidate documents
- **SciRepEval** as the primary benchmark for evaluating learned representations
- **SciDocs** only as a secondary benchmark for comparison with prior work
- **ACL Anthology** for smaller-scale prototyping or early-stage experiments
- **PeerRead** only if the project is extended toward peer-review or acceptance prediction tasks
