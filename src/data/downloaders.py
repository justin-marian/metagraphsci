"""
Stable public download entry points for MetaGraphSci datasets.

This module avoids importing heavier OpenAlex-query implementations at module
load time while still exposing a consistent import path for external callers.
It also contains lightweight benchmark downloaders for Planetoid, OGBN-Arxiv,
FoRC2025, and REST-based OpenAlex slices.
"""

from __future__ import annotations

import json
import shutil
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.datasets import Planetoid

from .constants import FORC2025_URL
from .download_utils import (
    download_file, empty_citations, ensure_required_columns, extract_zip,
    find_candidates, mask_to_split, read_table, save_benchmark_config,
    save_dataset_bundle, save_frame)

OPENALEX_API_URL = "https://api.openalex.org/works"
OPENALEX_SELECT = (
    "id,title,abstract_inverted_index,primary_location,authorships,"
    "publication_year,primary_topic,referenced_works")

PAGES_PER_FLUSH = 5  # ~1000 rows per flush: 5 pages * 200 per_page.
MAX_RETRIES = 5
RETRY_BACKOFF_CAP_SEC = 30.0


def download_openalex_query(
    out_dir: str, config_template: str | Path,
    query: str | None = None, max_papers: int = 5_000,
    from_year: int | None = None, to_year: int | None = None,
    include_citations: bool = True, expand_citations: bool = True,
    use_llm_fallback: bool = False, llm_api_key: str | None = None,
    papers_per_class: int | None = None,
    classes: list[str] | None = None,
    domain: str = "cs_ai") -> None:
    """
    Delegate OpenAlex query construction to the optional paper_labeler implementation.

    The local import keeps optional dependencies and startup cost out of code
    paths that only need the stable public interface.
    """
    from paper_labeler import download_openalex_query as impl

    impl(
        out_dir=out_dir, config_template=config_template, query=query,
        max_papers=max_papers, from_year=from_year, to_year=to_year,
        include_citations=include_citations, expand_citations=expand_citations,
        use_llm_fallback=use_llm_fallback, llm_api_key=llm_api_key,
        papers_per_class=papers_per_class, classes=classes, domain=domain)


def download_ogbn_arxiv(out_dir: str | Path, config_template: str | Path) -> None:
    """Download OGBN-Arxiv and export it in the project layout."""
    # OGBN-Arxiv provides explicit temporal splits. Preserve them to avoid
    # leakage from future papers into past-paper prediction.
    out = Path(out_dir)
    dataset = NodePropPredDataset(name="ogbn-arxiv", root=str(out / "raw"))
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
        "authors": [""] * num_nodes, "year": years, "original_split": split})

    edges = graph["edge_index"]
    citations = pl.DataFrame({"source": edges[0].tolist(), "target": edges[1].tolist()})
    save_dataset_bundle("ogbn_arxiv", out, docs, config_template, citations)


def normalize_forc_documents(df: pl.DataFrame) -> pl.DataFrame:
    """Rename and clean FoRC document fields so they match the shared schema."""
    # FoRC releases may vary column names between versions. Candidate lists make
    # ingestion resilient without hardcoding one exact archive schema.
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

    # Multi-label lists do not fit CrossEntropyLoss; use the first label as the
    # stable single-label target for this benchmark path.
    if "label" in out.columns and isinstance(out.schema["label"], pl.List):
        out = out.with_columns(pl.col("label").list.first().alias("label"))

    return ensure_required_columns(out)


def normalize_forc_citations(df: pl.DataFrame) -> pl.DataFrame:
    """Rename FoRC citation columns to the standard edge layout."""
    lower_map = {column.lower(): column for column in df.columns}
    src_col = lower_map.get("source") or lower_map.get("citing") or lower_map.get("from") or lower_map.get("src")
    tgt_col = lower_map.get("target") or lower_map.get("cited") or lower_map.get("to") or lower_map.get("dst")

    if not src_col or not tgt_col:
        raise ValueError("Could not find source/target columns in FoRC citations file")

    return df.rename({src_col: "source", tgt_col: "target"}).select(["source", "target"])


def download_forc2025(out_dir: str | Path, config_template: str | Path) -> None:
    """Download FoRC2025, detect relevant tables, and export normalized outputs."""
    out = Path(out_dir)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "FoRC2025_data.zip"
    extract_dir = raw_dir / "FoRC2025_data"

    if not zip_path.exists():
        download_file(FORC2025_URL, zip_path)
    if not extract_dir.exists():
        extract_zip(zip_path, extract_dir)

    # Do not hardcode ZIP filenames. If automatic resolution fails, write a
    # manifest so the researcher can inspect candidates and map files manually.
    doc_cands = [
        path for path in find_candidates(
            extract_dir, ("*document*.csv", "*documents*.csv", "*paper*.csv", "*.parquet", "*.jsonl"))
        if "citation" not in path.name.lower() and "edge" not in path.name.lower()]
    cit_cands = [
        path for path in find_candidates(
            extract_dir, ("*citation*.csv", "*citations*.csv", "*edge*.csv", "*graph*.csv", "*.parquet", "*.jsonl"))
        if any(key in path.name.lower() for key in ("citation", "edge", "graph"))]

    manifest = {
        "download_url": FORC2025_URL, "zip_path": str(zip_path), "extract_dir": str(extract_dir),
        "document_candidates": [str(path) for path in doc_cands],
        "citation_candidates": [str(path) for path in cit_cands]}
    (out / "forc_manifest.json").write_text(json.dumps(manifest, indent=2))

    if not doc_cands or not cit_cands:
        save_benchmark_config("forc4cl", out, config_template)
        (out / "README_forc4cl.json").write_text(json.dumps({
            "status": "downloaded_but_manual_mapping_needed",
            "message": (
                "FoRC2025 archive was downloaded and extracted, but document/citation "
                "files could not be mapped automatically. Inspect forc_manifest.json.")}, indent=2))
        return

    documents = normalize_forc_documents(read_table(doc_cands[0]))
    citations = normalize_forc_citations(read_table(cit_cands[0]))
    save_dataset_bundle("forc4cl", out, documents, config_template, citations)


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Rebuild plain-text abstract from OpenAlex inverted-index encoding."""
    # OpenAlex stores abstracts as {word: [positions]}. Invert the mapping so
    # downstream tokenizers receive normal ordered text.
    if not inverted_index:
        return ""

    positions: dict[int, str] = {}
    for word, indices in inverted_index.items():
        for idx in indices:
            positions[int(idx)] = word

    return " ".join(positions[idx] for idx in sorted(positions))


def openalex_id_to_int(oa_id: str) -> int:
    """Convert an OpenAlex Work URL/ID, e.g. .../W2741809807, to integer doc_id."""
    return int(oa_id.rsplit("/", 1)[-1].lstrip("W"))


def openalex_label(work: dict[str, Any], label_field: str) -> str:
    """Extract the configured taxonomy slot from OpenAlex primary_topic."""
    # Topics are hierarchical: topic ⊂ subfield ⊂ field ⊂ domain. The caller
    # chooses granularity for coarse or fine-grained experiments.
    topic = work.get("primary_topic") or {}

    if label_field == "topic":
        return topic.get("display_name") or ""
    if label_field == "subfield":
        return (topic.get("subfield") or {}).get("display_name") or ""
    if label_field == "domain":
        return (topic.get("domain") or {}).get("display_name") or ""

    return (topic.get("field") or {}).get("display_name") or ""


def format_duration(seconds: float) -> str:
    """Format a duration as a compact human-readable string."""
    if seconds < 0 or seconds != seconds:
        return "?"

    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"

    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def http_get_with_retry(url: str) -> dict[str, Any]:
    """GET a JSON endpoint with exponential backoff for transient failures."""
    # Retry network blips, OpenAlex 5xx, and 429s. Hard 4xx errors are surfaced
    # immediately because retrying malformed filters only delays the real error.
    delay = 1.0
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"OpenAlex API error {exc.code}: {body}") from exc
            last_exc = exc
        except (urllib.error.URLError, socket.timeout, json.JSONDecodeError) as exc:
            last_exc = exc

        if attempt < MAX_RETRIES - 1:
            print(f"[openalex]   transient error ({type(last_exc).__name__}); retry {attempt + 1}/{MAX_RETRIES} in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2.0, RETRY_BACKOFF_CAP_SEC)

    raise RuntimeError(f"OpenAlex API failed after {MAX_RETRIES} retries: {last_exc}")


def normalize_work(work: dict[str, Any], label_field: str) -> tuple[dict[str, Any] | None, list[tuple[int, int]]]:
    """Extract the schema-conformant doc row and outgoing citation edges from one work."""
    oa_id = work.get("id") or ""
    if not oa_id:
        return None, []

    try:
        doc_id = openalex_id_to_int(oa_id)
    except ValueError:
        return None, []

    label = openalex_label(work, label_field)
    if not label:
        return None, []

    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    authors = [
        (author.get("author") or {}).get("display_name") or ""
        for author in work.get("authorships") or []]
    authors = [author for author in authors if author]

    row = {
        "doc_id": doc_id, "title": work.get("title") or "",
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "venue": source.get("display_name") or "",
        "publisher": source.get("host_organization_name") or "",
        "authors": authors, "year": work.get("publication_year"), "label": label}

    edges: list[tuple[int, int]] = []
    for ref in work.get("referenced_works") or []:
        try:
            edges.append((doc_id, openalex_id_to_int(ref)))
        except ValueError:
            continue

    return row, edges


def flush_part(parts_dir: Path, part_idx: int, doc_buf: list[dict[str, Any]], cit_buf: list[tuple[int, int]]) -> None:
    """Spill buffers to numbered part files so peak RAM stays bounded."""
    parts_dir.mkdir(parents=True, exist_ok=True)

    if doc_buf:
        docs = pl.DataFrame({
            "doc_id": pl.Series([row["doc_id"] for row in doc_buf], dtype=pl.Int64),
            "title": [row["title"] for row in doc_buf],
            "abstract": [row["abstract"] for row in doc_buf],
            "venue": [row["venue"] for row in doc_buf],
            "publisher": [row["publisher"] for row in doc_buf],
            "authors": pl.Series([row["authors"] for row in doc_buf], dtype=pl.List(pl.String)),
            "year": pl.Series([row["year"] for row in doc_buf], dtype=pl.Int64),
            "label": [row["label"] for row in doc_buf]})
        docs.write_parquet(parts_dir / f"documents_{part_idx:04d}.parquet")

    if cit_buf:
        cits = pl.DataFrame({
            "source": pl.Series([src for src, _ in cit_buf], dtype=pl.Int64),
            "target": pl.Series([dst for _, dst in cit_buf], dtype=pl.Int64)})
        cits.write_parquet(parts_dir / f"citations_{part_idx:04d}.parquet")


def save_progress(progress_path: Path, state: dict[str, Any]) -> None:
    """Atomically persist resume state so a crash never leaves a half-written file."""
    tmp = progress_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(progress_path)


def load_progress(progress_path: Path, filter_str: str, label_field: str) -> dict[str, Any] | None:
    """Validate and return prior progress state, or None for a fresh start."""
    # Refuse to resume incompatible slices rather than silently mixing filters.
    if not progress_path.exists():
        return None

    state = json.loads(progress_path.read_text())
    if state.get("filter") != filter_str or state.get("label_field") != label_field:
        raise RuntimeError(
            f"Existing partial download at {progress_path.parent} used different "
            f"--oa_filter or --oa_label_field. Re-run with the original parameters "
            f"(filter={state.get('filter')!r}, label_field={state.get('label_field')!r}) "
            f"or delete {progress_path.parent} to start fresh.")

    return state


def finalize_openalex(out: Path, parts_dir: Path, config_template: str | Path) -> tuple[int, int]:
    """Concatenate part files and filter citations to the intra-slice subgraph."""
    # rglob handles serial and per-worker directories uniformly.
    doc_files = sorted(parts_dir.rglob("documents_*.parquet"))
    if not doc_files:
        raise RuntimeError(f"No document parts found in {parts_dir}.")

    documents = pl.concat([pl.read_parquet(path) for path in doc_files]).unique(subset=["doc_id"])
    documents = ensure_required_columns(documents)

    cit_files = sorted(parts_dir.rglob("citations_*.parquet"))
    if cit_files:
        valid_ids = set(documents["doc_id"].to_list())
        citations = (
            pl.concat([pl.read_parquet(path) for path in cit_files])
            .filter(pl.col("source").is_in(valid_ids) & pl.col("target").is_in(valid_ids)).unique())
    else:
        citations = empty_citations()

    save_frame(documents, out / "documents.parquet")
    save_frame(citations, out / "citations.parquet")
    save_benchmark_config("openalex", out, config_template)

    # Clean up only after final outputs are safely written, preserving resumability on failure.
    shutil.rmtree(parts_dir, ignore_errors=True)
    for progress_file in out.glob("_progress_*.json"):
        progress_file.unlink(missing_ok=True)
    (out / "_progress.json").unlink(missing_ok=True)

    return documents.height, citations.height


def stream_worker(out: Path, worker_id: str, filter_str: str, label_field: str, max_works: int, email: str, log_prefix: str) -> int:
    """Run one streaming OpenAlex worker and return total collected papers."""
    # Each worker owns its _parts subdir and progress file, so workers can crash
    # and resume independently without coordination.
    parts_dir = out / "_parts" / worker_id
    progress_path = out / f"_progress_{worker_id}.json"
    full_filter = f"has_abstract:true,{filter_str}" if filter_str else "has_abstract:true"

    state = load_progress(progress_path, filter_str, label_field)
    if state is not None:
        cursor = state["cursor"]
        collected = state["collected"]
        part_idx = state["next_part_idx"]
        print(f"{log_prefix} resuming: {collected} papers already on disk")
        if collected >= max_works:
            print(f"{log_prefix} already at quota ({collected} >= {max_works})")
            return collected
    else:
        cursor = "*"
        collected = 0
        part_idx = 0
        print(f"{log_prefix} starting fresh (filter='{full_filter}', max={max_works})")

    base_params: dict[str, str | int] = {
        "filter": full_filter, "per_page": 200, "select": OPENALEX_SELECT}
    if email:
        base_params["mailto"] = email

    doc_buf: list[dict[str, Any]] = []
    cit_buf: list[tuple[int, int]] = []
    pages_since_flush = 0
    session_start = time.monotonic()
    session_start_collected = collected

    while collected < max_works and cursor is not None:
        params = {**base_params, "cursor": cursor}
        page = http_get_with_retry(f"{OPENALEX_API_URL}?{urllib.parse.urlencode(params)}")
        batch = page.get("results") or []

        if not batch:
            break

        remaining = max_works - collected
        for work in batch[:remaining]:
            row, edges = normalize_work(work, label_field)
            if row is None:
                continue
            doc_buf.append(row)
            cit_buf.extend(edges)
            collected += 1

        cursor = (page.get("meta") or {}).get("next_cursor")
        pages_since_flush += 1

        elapsed = time.monotonic() - session_start
        session_collected = collected - session_start_collected
        if session_collected > 0 and elapsed > 0:
            rate = session_collected / elapsed
            eta = (max_works - collected) / rate if rate > 0 else float("inf")
            print(
                f"{log_prefix}   collected {collected} / {max_works}  "
                f"({rate:.1f} papers/s, elapsed {format_duration(elapsed)}, "
                f"eta {format_duration(eta)})")
        else:
            print(f"{log_prefix}   collected {collected} / {max_works}")

        if pages_since_flush >= PAGES_PER_FLUSH or collected >= max_works or cursor is None:
            flush_part(parts_dir, part_idx, doc_buf, cit_buf)
            part_idx += 1
            doc_buf.clear()
            cit_buf.clear()
            pages_since_flush = 0
            save_progress(progress_path, {
                "filter": filter_str, "label_field": label_field, "cursor": cursor,
                "collected": collected, "next_part_idx": part_idx})

    return collected


def partition_years(year_min: int, year_max: int, workers: int) -> list[tuple[int, int]]:
    """Split ``[year_min, year_max]`` into contiguous non-overlapping year ranges."""
    # Equal-width buckets are simple and disjoint. OpenAlex density skews recent,
    # but this avoids a pre-flight density-counting round trip.
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if year_max < year_min:
        raise ValueError(f"year_max ({year_max}) must be >= year_min ({year_min})")

    span = year_max - year_min + 1
    if workers > span:
        raise ValueError(f"--oa_workers ({workers}) cannot exceed year span ({span})")

    width = span / workers
    ranges: list[tuple[int, int]] = []
    for idx in range(workers):
        lo = year_min + int(idx * width)
        hi = year_min + int((idx + 1) * width) - 1 if idx < workers - 1 else year_max
        ranges.append((lo, hi))

    return ranges


def download_openalex(
    out_dir: str | Path, config_template: str | Path, *,
    filter_str: str = "primary_topic.field.id:17",
    max_works: int = 50_000, email: str = "",
    label_field: str = "field", workers: int = 1,
    year_min: int = 2000, year_max: int | None = None) -> None:
    """
    Fetch a slice of OpenAlex via REST API and export it in the project layout.

    Serial mode uses one cursor-paginated stream. Parallel mode partitions the
    year window into disjoint ranges and runs independent worker threads.
    """
    # Year-partitioned parallelism is server-side, disjoint, and I/O-bound, so
    # threads are sufficient and cheaper than processes.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts_dir = out / "_parts"

    if workers <= 1:
        total = stream_worker(out, "w0", filter_str, label_field, max_works, email, "[openalex]")
    else:
        year_max_eff = year_max if year_max is not None else datetime.now().year
        ranges = partition_years(year_min, year_max_eff, workers)
        per_worker_max = -(-max_works // workers)
        print(f"[openalex] parallel: {workers} workers, year ranges {ranges}, ~{per_worker_max} papers/worker")

        def _worker_filter(lo: int, hi: int) -> str:
            year_clause = f"from_publication_date:{lo}-01-01,to_publication_date:{hi}-12-31"
            return f"{filter_str},{year_clause}" if filter_str else year_clause

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    stream_worker, out, f"w{idx}", _worker_filter(lo, hi),
                    label_field, per_worker_max, email, f"[openalex/w{idx}:{lo}-{hi}]"): idx
                for idx, (lo, hi) in enumerate(ranges)}
            total = sum(future.result() for future in as_completed(futures))

    if total == 0:
        raise RuntimeError("OpenAlex returned no usable results; check your --oa_filter argument.")

    n_docs, n_cits = finalize_openalex(out, parts_dir, config_template)
    print(f"[openalex] wrote {n_docs} documents and {n_cits} citations to {out}")


def download_planetoid_dataset(name: str, out_dir: str | Path, config_template: str | Path) -> None:
    """Download a Planetoid benchmark and convert it to the shared schema."""
    # Deconstruct PyG objects into universal documents + citations tables so the
    # training pipeline uses one ingest path across PyG, OGB, OpenAlex, and CSVs.
    out = Path(out_dir)
    dataset = Planetoid(root=str(out / "raw"), name=name)
    data = dataset[0]

    docs = pl.DataFrame({
        "doc_id": list(range(data.num_nodes)), "title": [""] * data.num_nodes,
        "abstract": [""] * data.num_nodes, "label": data.y.cpu().numpy().tolist(),
        "venue": [name] * data.num_nodes, "publisher": ["Planetoid"] * data.num_nodes,
        "authors": [""] * data.num_nodes, "year": [None] * data.num_nodes,
        "original_split": mask_to_split(data.train_mask.cpu(), data.val_mask.cpu(), data.test_mask.cpu())})

    edge_index = data.edge_index.cpu().numpy()
    citations = pl.DataFrame({"source": edge_index[0].tolist(), "target": edge_index[1].tolist()})
    save_dataset_bundle("cora" if name.lower() == "cora" else "pubmed", out, docs, config_template, citations)
