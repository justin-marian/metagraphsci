from __future__ import annotations

import argparse
import json
import yaml
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import polars as pl
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import NodePropPredDataset

"""Download and normalize supported benchmarks for MetaGraphSci.

This file isolates dataset-specific quirks from the rest of the codebase. Each
download routine converts an external dataset into the same internal layout used
by the training pipeline and writes a ready-to-run config next to the exported files.

- fetch supported benchmarks from their original sources,
- map heterogeneous fields onto the shared project schema,
- export normalized documents and citations tables,
- generate a matching config so experiments can start immediately.
"""


REQUIRED_COLUMNS = ["doc_id", "title", "abstract", "venue", "publisher", "authors", "year"]
BENCHMARK_DEFAULTS = {
    "generic": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "random"},
    "cora": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "random"},
    "pubmed": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "random"},
    "ogbn_arxiv": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "time"},
    "forc4cl": {"label_column": "label", "source_col": "source", "target_col": "target", "split_strategy": "time"}}
DOCUMENT_COLUMNS = REQUIRED_COLUMNS + ["label"]
FORC2025_URL = "https://zenodo.org/records/14901529/files/FoRC2025_data.zip?download=1"


def save_frame(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.write_parquet(path)
    else:
        df.write_csv(path)


def empty_citations() -> pl.DataFrame:
    return pl.DataFrame({"source": pl.Series(name="source", values=[], dtype=pl.Int64), "target": pl.Series(name="target", values=[], dtype=pl.Int64)})


def ensure_required_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Fill missing columns and reorder fields to match the project schema."""
    defaults: dict[str, Any] = {"title": "", "abstract": "", "venue": "", "publisher": "", "authors": "", "year": None}
    out = df

    if "doc_id" not in out.columns:
        out = out.with_row_index("doc_id")

    for column in REQUIRED_COLUMNS:
        if column not in out.columns:
            out = out.with_columns(pl.lit(defaults[column]).alias(column))

    if "label" not in out.columns:
        raise ValueError("documents frame must include a 'label' column")

    # Reorder columns so required fields come first, followed by any extras in their original order.
    ordered = [column for column in DOCUMENT_COLUMNS if column in out.columns]
    extra = [column for column in out.columns if column not in ordered]
    return out.select(ordered + extra)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config template not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def save_benchmark_config(dataset_name: str, out_dir: Path, config_template: str | Path) -> None:
    """Write a benchmark-specific config next to the exported dataset files."""
    cfg = load_yaml(config_template)
    cfg.setdefault("project", {})
    cfg.setdefault("data", {})

    # Populate config fields that identify the benchmark and its data files, as well as any defaults needed by the pipeline.
    cfg["project"]["benchmark"] = dataset_name
    cfg["project"]["run_name"] = f"MetaGraphSci_{dataset_name}"

    # Assumes documents.csv, citations.csv, and baselines.csv (if needed) 
    # will be placed in the output directory. Adjust if your layout differs.
    cfg["data"]["documents"] = str(out_dir / "documents.csv")
    cfg["data"]["citations"] = str(out_dir / "citations.csv")
    cfg["data"]["baselines"] = str(out_dir / "baselines.csv")

    # Add any benchmark-specific defaults for the pipeline, such as label column names or split strategies.
    for key, value in BENCHMARK_DEFAULTS[dataset_name].items():
        cfg["data"][key] = value

    # Write the updated config back to disk in the output directory.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def save_dataset_bundle(
    dataset_name: str, out_dir: str | Path, documents: pl.DataFrame,
    config_template: str | Path, citations: pl.DataFrame | None = None) -> None:
    """Export normalized tables and generate the config used by the pipeline."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    docs = ensure_required_columns(documents)
    cits = citations if citations is not None else empty_citations()

    save_frame(docs, out / "documents.csv")
    save_frame(cits, out / "citations.csv")
    save_benchmark_config(dataset_name, out, config_template)


def mask_to_split(train_mask, val_mask, test_mask) -> list[str]:
    """Convert boolean split masks into human-readable split names."""
    split: list[str] = []
    for is_train, is_val, is_test in zip(train_mask.tolist(), val_mask.tolist(), test_mask.tolist()):
        if is_train:
            split.append("train")
        elif is_val:
            split.append("val")
        elif is_test:
            split.append("test")
        else:
            split.append("unassigned")
    return split


def download_planetoid_dataset(name: str, out_dir: str, config_template: str | Path) -> None:
    """Download a Planetoid benchmark and convert it to the shared schema."""
    dataset = Planetoid(root=str(Path(out_dir) / "raw"), name=name)
    data = dataset[0]
    docs = pl.DataFrame({
        "doc_id": list(range(data.num_nodes)),
        "title": [""] * data.num_nodes,
        "abstract": [""] * data.num_nodes,
        "label": data.y.cpu().numpy().tolist(),
        "venue": [name] * data.num_nodes,
        "publisher": ["Planetoid"] * data.num_nodes,
        "authors": [""] * data.num_nodes,
        "year": [None] * data.num_nodes,
        "original_split": mask_to_split(data.train_mask.cpu(), data.val_mask.cpu(), data.test_mask.cpu())
    })

    edge_index = data.edge_index.cpu().numpy()
    citations = pl.DataFrame({"source": edge_index[0].tolist(), "target": edge_index[1].tolist()})

    dataset_name = "cora" if name.lower() == "cora" else "pubmed"
    save_dataset_bundle(dataset_name, out_dir, docs, config_template, citations)


def download_ogbn_arxiv(out_dir: str, config_template: str | Path) -> None:
    """Download OGBN-Arxiv and export it in the project layout."""
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
        "doc_id": list(range(num_nodes)), 
        "title": [""] * num_nodes, "abstract": [""] * num_nodes,
        "label": labels.reshape(-1).tolist(),
        "venue": ["arXiv"] * num_nodes, "publisher": ["OGB"] * num_nodes, 
        "authors": [""] * num_nodes, "year": years, "original_split": split
    })

    edges = graph["edge_index"]
    citations = pl.DataFrame({"source": edges[0].tolist(), "target": edges[1].tolist()})
    save_dataset_bundle("ogbn_arxiv", out_dir, docs, config_template, citations)


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, path.open("wb") as file_handle:
        shutil.copyfileobj(response, file_handle)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(out_dir)


def find_candidates(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    """Find candidate files matching a set of glob patterns."""
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.rglob(pattern))
    return sorted({path for path in found if path.is_file()})


def read_table(path: Path) -> pl.DataFrame:
    if path.suffix == ".csv":
        return pl.read_csv(path)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix == ".jsonl":
        return pl.read_ndjson(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return pl.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("rows", "data", "documents", "records"):
                if key in payload and isinstance(payload[key], list):
                    return pl.DataFrame(payload[key])
        raise ValueError(f"Unsupported JSON table structure: {path}")
    raise ValueError(f"Unsupported table format: {path.suffix}")


def normalize_forc_documents(df: pl.DataFrame) -> pl.DataFrame:
    """Rename and clean FoRC document fields so they match the shared schema."""
    rename_map: dict[str, str] = {}
    lower_map = {column.lower(): column for column in df.columns}
    candidate_sets = {
        "doc_id": ["doc_id", "id", "paper_id", "publication_id"],
        "title": ["title", "paper_title"],
        "abstract": ["abstract", "summary", "paper_abstract"],
        "venue": ["venue", "booktitle", "journal"],
        "publisher": ["publisher"],
        "authors": ["authors", "author", "author_names"],
        "year": ["year", "publication_year"],
        "label": ["label", "labels", "topic", "field", "class"]}

    for target, candidates in candidate_sets.items():
        for candidate in candidates:
            if candidate in lower_map:
                rename_map[lower_map[candidate]] = target
                break

    out = df.rename(rename_map)
    if "labels" in out.columns and "label" not in out.columns:
        out = out.rename({"labels": "label"})
    if "label" in out.columns and out.schema["label"] == pl.List:
        out = out.with_columns(pl.col("label").list.first().alias("label"))
    return ensure_required_columns(out)


def normalize_forc_citations(df: pl.DataFrame) -> pl.DataFrame:
    """Rename FoRC citation columns to the standard edge layout used elsewhere."""
    lower_map = {column.lower(): column for column in df.columns}
    source_col = lower_map.get("source") or lower_map.get("citing") or lower_map.get("from") or lower_map.get("src")
    target_col = lower_map.get("target") or lower_map.get("cited") or lower_map.get("to") or lower_map.get("dst")

    if not source_col or not target_col:
        raise ValueError("Could not find source/target columns in FoRC citations file")

    out = df.rename({source_col: "source", target_col: "target"})
    return out.select(["source", "target"])


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

    doc_candidates = find_candidates(extract_dir, ("*document*.csv", "*documents*.csv", "*paper*.csv", "*publication*.csv", "*.parquet", "*.jsonl"))
    cit_candidates = find_candidates(extract_dir, ("*citation*.csv", "*citations*.csv", "*edge*.csv", "*graph*.csv", "*.parquet", "*.jsonl"))

    doc_candidates = [path for path in doc_candidates if "citation" not in path.name.lower() and "edge" not in path.name.lower()]
    cit_candidates = [path for path in cit_candidates if any(key in path.name.lower() for key in ("citation", "edge", "graph"))]

    manifest = {
        "download_url": FORC2025_URL, "zip_path": str(zip_path), "extract_dir": str(extract_dir),
        "document_candidates": [str(path) for path in doc_candidates],
        "citation_candidates": [str(path) for path in cit_candidates]}
    (out / "forc_manifest.json").write_text(json.dumps(manifest, indent=2))

    if not doc_candidates or not cit_candidates:
        save_benchmark_config("forc4cl", out, config_template)
        note = {
            "status": "downloaded_but_manual_mapping_needed",
            "message": (
                "FoRC 2025 archive downloaded and extracted, but document/citation files "
                "could not be mapped automatically. Inspect forc_manifest.json and raw/FoRC2025_data.")}
        (out / "README_forc4cl.json").write_text(json.dumps(note, indent=2))
        return

    documents = normalize_forc_documents(read_table(doc_candidates[0]))
    citations = normalize_forc_citations(read_table(cit_candidates[0]))
    save_dataset_bundle("forc4cl", out, documents, config_template, citations)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset download commands."""
    parser = argparse.ArgumentParser(description="Download datasets used by the MetaGraphSci analysis pipeline.")
    parser.add_argument("--dataset", required=True, choices=["cora", "pubmed", "ogbn_arxiv", "forc4cl"])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config_template", default="config.yaml", help="Base YAML config template to clone and update.")
    return parser.parse_args()


def main() -> None:
    """Dispatch the requested dataset download routine."""
    args = parse_args()

    if args.dataset == "cora":
        download_planetoid_dataset("Cora", args.out_dir, args.config_template)
    elif args.dataset == "pubmed":
        download_planetoid_dataset("PubMed", args.out_dir, args.config_template)
    elif args.dataset == "ogbn_arxiv":
        download_ogbn_arxiv(args.out_dir, args.config_template)
    elif args.dataset == "forc4cl":
        download_forc2025(args.out_dir, args.config_template)


if __name__ == "__main__":
    main()
