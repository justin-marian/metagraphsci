import json
from pathlib import Path
import polars as pl
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import NodePropPredDataset

from constants import FORC2025_URL
from download_utils import (
    save_dataset_bundle, save_benchmark_config, 
    mask_to_split, find_candidates, 
    download_file, extract_zip,
    read_table, ensure_required_columns)


def download_planetoid_dataset(name: str, out_dir: str, config_template: str | Path) -> None:
    """Download a Planetoid benchmark and convert it to the shared schema."""
    
    # Graph Deconstruction & Homogenization
    # PyTorch Geometric provides Planetoid as a ready-to-use object. However, 
    # feeding this directly into the pipeline couples the model to PyG's specific format. 
    # Instead, intentionally deconstruct the PyG object into our universal tabular 
    # schema (documents + citations). This guarantees the training pipeline only needs 
    # one ingest logic, whether the data came from PyG, OGB, or a raw CSV download.
    dataset = Planetoid(root=str(Path(out_dir) / "raw"), name=name)
    data = dataset[0]
    
    docs = pl.DataFrame({
        "doc_id": list(range(data.num_nodes)), "title": [""] * data.num_nodes, 
        "abstract": [""] * data.num_nodes, "label": data.y.cpu().numpy().tolist(),
        "venue": [name] * data.num_nodes, "publisher": ["Planetoid"] * data.num_nodes,
        "authors": [""] * data.num_nodes, "year": [None] * data.num_nodes,
        "original_split": mask_to_split(data.train_mask.cpu(), data.val_mask.cpu(), data.test_mask.cpu())
    })

    edge_index = data.edge_index.cpu().numpy()
    citations = pl.DataFrame({"source": edge_index[0].tolist(), "target": edge_index[1].tolist()})
    save_dataset_bundle("cora" if name.lower() == "cora" else "pubmed", out_dir, docs, config_template, citations)


def download_ogbn_arxiv(out_dir: str, config_template: str | Path) -> None:
    """Download OGBN-Arxiv and export it in the project layout."""
    
    # ARCHITECTURAL DECISION: Temporal Data Leakage Prevention
    # Unlike Cora/PubMed, the OGB benchmark provides explicit train/val/test splits 
    # based on time (e.g., train on papers before 2017, test on 2019). We extract 
    # and strictly preserve this split mapping. If we applied a random split here, 
    # the model could use future citation structures to predict past papers, 
    # ruining the validity of the benchmark.
    dataset = NodePropPredDataset(name="ogbn-arxiv", root=str(Path(out_dir) / "raw"))
    graph, labels = dataset[0]
    split_idx = dataset.get_idx_split()
    num_nodes = int(graph["num_nodes"])

    split = ["unassigned"] * num_nodes
    for idx in split_idx["train"].reshape(-1).tolist(): 
        split[int(idx)] = "train"
    for idx in split_idx["valid"].reshape(-1).tolist(): 
        split[int(idx)] = "val"
    for idx in split_idx["test"].reshape(-1).tolist(): 
        split[int(idx)] = "test"

    years = graph["node_year"].reshape(-1).tolist() if "node_year" in graph else [None] * num_nodes
    docs = pl.DataFrame({
        "doc_id": list(range(num_nodes)), "title": [""] * num_nodes, 
        "abstract": [""] * num_nodes, "label": labels.reshape(-1).tolist(),
        "venue": ["arXiv"] * num_nodes, "publisher": ["OGB"] * num_nodes, 
        "authors": [""] * num_nodes, "year": years, "original_split": split
    })

    edges = graph["edge_index"]
    citations = pl.DataFrame({"source": edges[0].tolist(), "target": edges[1].tolist()})
    save_dataset_bundle("ogbn_arxiv", out_dir, docs, config_template, citations)


def normalize_forc_documents(df: pl.DataFrame) -> pl.DataFrame:
    """Rename and clean FoRC document fields so they match the shared schema."""
    
    # ARCHITECTURAL DECISION: Defensive Schema Mapping (Fuzzy Ingestion)
    # Academic datasets distributed as raw files frequently change column names 
    # between versions (e.g., "paper_id" in v1 becomes "publication_id" in v2). 
    # By defining lists of candidate column names and falling back iteratively, 
    # we make the data pipeline resilient to upstream formatting changes.
    rename_map: dict[str, str] = {}
    lower_map = {c.lower(): c for c in df.columns}
    candidate_sets = {
        "doc_id": ["doc_id", "id", "paper_id", "publication_id"],
        "title": ["title", "paper_title"],
        "abstract": ["abstract", "summary", "paper_abstract"],
        "venue": ["venue", "booktitle", "journal"],
        "publisher": ["publisher"],
        "authors": ["authors", "author", "author_names"],
        "year": ["year", "publication_year"],
        "label": ["label", "labels", "topic", "field", "class"]
    }

    for target, candidates in candidate_sets.items():
        for candidate in candidates:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = target
                break

    out = df.rename(rename_map)
    if "labels" in out.columns and "label" not in out.columns: 
        out = out.rename({"labels": "label"})
    
    # Multi-label to Single-label Coercion
    # If the dataset provides a list of labels per document, it breaks standard 
    # CrossEntropyLoss. Coerce this to a single-label problem by taking the first 
    # element, prioritizing mathematical stability over full multi-class representation.
    if "label" in out.columns and out.schema["label"] == pl.List:
        out = out.with_columns(pl.col("label").list.first().alias("label"))
    return ensure_required_columns(out)


def normalize_forc_citations(df: pl.DataFrame) -> pl.DataFrame:
    """Rename FoRC citation columns to the standard edge layout used elsewhere."""
    lower_map = {c.lower(): c for c in df.columns}
    src_col = lower_map.get("source") or lower_map.get("citing") or lower_map.get("from") or lower_map.get("src")
    tgt_col = lower_map.get("target") or lower_map.get("cited") or lower_map.get("to") or lower_map.get("dst")

    if not src_col or not tgt_col: raise ValueError("Could not find source/target columns in FoRC citations file")
    return df.rename({src_col: "source", tgt_col: "target"}).select(["source", "target"])


def download_forc2025(out_dir: str, config_template: str | Path) -> None:
    """Download FoRC 2025, detect the relevant tables, and export normalized outputs."""
    out = Path(out_dir)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "FoRC2025_data.zip"
    extract_dir = raw_dir / "FoRC2025_data"

    if not zip_path.exists():
        download_file(FORC2025_URL, zip_path)
    if not extract_dir.exists(): 
        extract_zip(zip_path, extract_dir)

    # Automated File Discovery & Manifesting
    # We do not hardcode the expected filenames inside the ZIP. 
    # Instead, we use regex/glob matching to find files containing "document" or "citation". 
    # If automatic resolution fails or is ambiguous, we write a manifest.json. This acts 
    # as an "eject button" allowing the researcher to manually inspect what went wrong 
    # rather than failing silently or processing the wrong file.
    doc_cands = [p for p in find_candidates(
        extract_dir, 
        ("*document*.csv", "*documents*.csv", "*paper*.csv", "*.parquet", "*.jsonl")
    ) if "citation" not in p.name.lower() and "edge" not in p.name.lower()]
    cit_cands = [p for p in find_candidates(
        extract_dir, 
        ("*citation*.csv", "*citations*.csv", "*edge*.csv", "*graph*.csv", "*.parquet", "*.jsonl")
    ) if any(k in p.name.lower() for k in ("citation", "edge", "graph"))]

    manifest = {
        "download_url": FORC2025_URL, 
        "zip_path": str(zip_path), 
        "extract_dir": str(extract_dir),
        "document_candidates": [str(p) for p in doc_cands], 
        "citation_candidates": [str(p) for p in cit_cands]
    }
    (out / "forc_manifest.json").write_text(json.dumps(manifest, indent=2))

    if not doc_cands or not cit_cands:
        save_benchmark_config("forc4cl", out, config_template)
        (out / "README_forc4cl.json").write_text(json.dumps({
            "status": "downloaded_but_manual_mapping_needed",
            "message": 
                ("FoRC 2025 archive download+extract,"
                "but document/citation files could not be mapped automatically."
                "Inspect forc_manifest.json.")
        }, indent=2))
        return

    documents = normalize_forc_documents(read_table(doc_cands[0]))
    citations = normalize_forc_citations(read_table(cit_cands[0]))
    save_dataset_bundle("forc4cl", out, documents, config_template, citations)
