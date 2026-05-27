"""Build an OOD holdout slice of OpenAlex AI papers for inference-only evaluation.

Downloads recent CS/AI papers via the OpenAlex REST API, filters out any whose
numeric OpenAlex id (== doc_id in the training parquet) already appears in
``data/openalex_ai/documents.parquet``, keeps only papers whose
``primary_topic.display_name`` matches a label in the original training label
set, and writes ``documents.parquet`` / ``citations.parquet`` / ``baselines.parquet``
to the requested output directory with the same schema as the training dataset.

Usage:
    python scripts/build_eval_holdout.py --n 100 --out data/openalex_ai_holdout100

Run from the repository root. Requires no GPU and no extra dependencies beyond
``polars`` and the standard library.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl
from loguru import logger

from src.data.constants import (
    OPENALEX_BASE, OPENALEX_HEADERS, OPENALEX_PAGE_SIZE, OPENALEX_RETRY_SLEEP_S,
    OPENALEX_RETRY_TIMEOUT_S, OPENALEX_SELECT, OPENALEX_SLEEP_S,
    OPENALEX_TIMEOUT_S, UNKNOWN_TOKEN, YEAR)

# Force unbuffered stderr so progress shows up under nohup/tee/redirected logs.
# Loguru's default sink goes to stderr but its buffering depends on the runtime;
# re-binding here also guarantees a sink exists even if some earlier import
# removed the defaults (src.* helpers do this in production runs).
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


# OpenAlex field id for "Artificial Intelligence" (level-1 field under "Computer Science").
# Filter scope used by the production AI dataset builder; matches the topic universe
# that produced the training labels in data/openalex_ai/documents.parquet.
AI_FIELD_ID = "fields/17"

DEFAULT_TRAIN_DOCS = "data/openalex_ai/documents.parquet"
DEFAULT_TRAIN_BASELINES = "data/openalex_ai/baselines.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=100, help="Number of holdout papers to keep (default: 100).")
    parser.add_argument("--out", type=Path, default=Path("data/openalex_ai_holdout100"),
                        help="Output directory for the holdout parquet files.")
    parser.add_argument("--train-docs", type=Path, default=Path(DEFAULT_TRAIN_DOCS),
                        help="Training documents.parquet, used to exclude already-seen ids and pin label set.")
    parser.add_argument("--train-baselines", type=Path, default=Path(DEFAULT_TRAIN_BASELINES),
                        help="Training baselines.parquet (schema source). Pass empty path to skip.")
    parser.add_argument("--from-year", type=int, default=2024,
                        help="Only fetch papers published from this year onwards (default: 2024 — likely after training crawl).")
    parser.add_argument("--max-fetch", type=int, default=2000,
                        help="Hard cap on API pages worth of candidates to scan (default: 2000 papers).")
    parser.add_argument("--mailto", type=str, default=None,
                        help="Optional polite-pool email for OpenAlex. Bumps rate limits.")
    return parser.parse_args()


def openalex_id_to_int(openalex_id: str) -> int | None:
    """Parse 'https://openalex.org/W12345' -> 12345; return None on malformed input."""
    if not openalex_id:
        return None
    tail = openalex_id.rsplit("/", 1)[-1].strip()
    if tail.startswith("W") and tail[1:].isdigit():
        return int(tail[1:])
    return None


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """OpenAlex returns abstracts as inverted indices; flatten back to text."""
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for token, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((int(idx), str(token)))
    positions.sort(key=lambda pair: pair[0])
    return " ".join(token for _, token in positions)


def extract_authors(authorships: list[dict[str, Any]] | None) -> list[str]:
    if not authorships:
        return []
    out: list[str] = []
    for entry in authorships:
        author = (entry or {}).get("author") or {}
        name = author.get("display_name")
        if name:
            out.append(str(name).strip())
    return out


def extract_venue(primary_location: dict[str, Any] | None) -> str:
    if not primary_location:
        return UNKNOWN_TOKEN
    source = primary_location.get("source") or {}
    name = source.get("display_name") or ""
    return str(name).strip() or UNKNOWN_TOKEN


def extract_publisher(primary_location: dict[str, Any] | None) -> str:
    if not primary_location:
        return UNKNOWN_TOKEN
    source = primary_location.get("source") or {}
    name = source.get("host_organization_name") or ""
    return str(name).strip() or UNKNOWN_TOKEN


def http_get(url: str, *, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=OPENALEX_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def fetch_with_retry(url: str) -> dict[str, Any]:
    """One automatic retry with a longer timeout — matches the downloader policy."""
    try:
        return http_get(url, timeout=OPENALEX_TIMEOUT_S)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as first:
        logger.warning("OpenAlex request failed ({}); retrying after {}s ...", first, OPENALEX_RETRY_SLEEP_S)
        time.sleep(OPENALEX_RETRY_SLEEP_S)
        return http_get(url, timeout=OPENALEX_RETRY_TIMEOUT_S)


def iter_openalex_works(
    *, from_year: int, mailto: str | None, max_papers: int,
) -> Iterable[dict[str, Any]]:
    """Cursor-paginate OpenAlex works filtered to AI papers from ``from_year`` onwards."""
    filter_expr = f"primary_topic.field.id:{AI_FIELD_ID},publication_year:>{from_year - 1}"
    params: dict[str, str] = {
        "filter": filter_expr,
        "select": OPENALEX_SELECT,
        "per-page": str(OPENALEX_PAGE_SIZE),
        "cursor": "*",
        "sort": "publication_date:desc"}
    if mailto:
        params["mailto"] = mailto

    fetched = 0
    page_no = 0
    while True:
        url = f"{OPENALEX_BASE}/works?{urllib.parse.urlencode(params)}"
        page_no += 1
        logger.info("OpenAlex page {} requesting up to {} works (total fetched so far: {})",
                    page_no, OPENALEX_PAGE_SIZE, fetched)
        page = fetch_with_retry(url)
        results = page.get("results") or []
        total_estimate = (page.get("meta") or {}).get("count")
        if page_no == 1 and total_estimate is not None:
            logger.info("OpenAlex reports {} works matching the filter; will scan up to {}",
                        total_estimate, max_papers)
        if not results:
            logger.warning("OpenAlex returned an empty page; stopping pagination.")
            return

        for work in results:
            yield work
            fetched += 1
            if fetched >= max_papers:
                return

        next_cursor = (page.get("meta") or {}).get("next_cursor")
        if not next_cursor:
            logger.info("No next_cursor returned; OpenAlex pagination exhausted at {} works.", fetched)
            return
        params["cursor"] = next_cursor
        time.sleep(OPENALEX_SLEEP_S)


def work_to_row(work: dict[str, Any]) -> dict[str, Any] | None:
    """Project an OpenAlex /works payload onto the training-parquet schema."""
    doc_id = openalex_id_to_int(work.get("id") or "")
    if doc_id is None:
        return None

    title = (work.get("title") or "").strip()
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    if not title and not abstract:
        return None

    primary_topic = work.get("primary_topic") or {}
    label = (primary_topic.get("display_name") or "").strip()
    if not label:
        return None

    year_raw = work.get("publication_year")
    try:
        year = int(year_raw) if year_raw is not None else int(YEAR)
    except (TypeError, ValueError):
        year = int(YEAR)

    referenced = [openalex_id_to_int(ref) for ref in (work.get("referenced_works") or [])]
    referenced_ids = [ref for ref in referenced if ref is not None]

    return {
        "doc_id": doc_id,
        "title": title,
        "abstract": abstract,
        "venue": extract_venue(work.get("primary_location")),
        "publisher": extract_publisher(work.get("primary_location")),
        "authors": extract_authors(work.get("authorships")),
        "year": year,
        "label": label,
        "_referenced_works": referenced_ids}


def main() -> None:
    args = parse_args()
    print(f"[build_eval_holdout] starting; out={args.out} n={args.n} from_year={args.from_year}", flush=True)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.train_docs.exists():
        raise FileNotFoundError(f"Training documents not found: {args.train_docs}")
    logger.info("Reading training documents from {}", args.train_docs)
    train_docs = pl.read_parquet(args.train_docs)
    train_ids = set(int(v) for v in train_docs["doc_id"].to_list())
    train_labels = set(str(v) for v in train_docs.drop_nulls("label")["label"].to_list())
    logger.info("Loaded training set: {} docs, {} unique labels, {} unique doc_ids",
                train_docs.height, len(train_labels), len(train_ids))

    selected_rows: list[dict[str, Any]] = []
    scanned = 0
    skipped_overlap = 0
    skipped_label_unknown = 0
    skipped_empty = 0

    for work in iter_openalex_works(
        from_year=args.from_year, mailto=args.mailto, max_papers=args.max_fetch
    ):
        scanned += 1
        row = work_to_row(work)
        if row is None:
            skipped_empty += 1
        elif row["doc_id"] in train_ids:
            skipped_overlap += 1
        elif row["label"] not in train_labels:
            skipped_label_unknown += 1
        else:
            selected_rows.append(row)

        if scanned % 200 == 0:
            logger.info("Scanned {} works; kept {} / {} so far (overlap={}, unknown_label={}, empty={})",
                        scanned, len(selected_rows), args.n, skipped_overlap,
                        skipped_label_unknown, skipped_empty)
        if len(selected_rows) >= args.n:
            break

    logger.info("Selection summary: kept={} need={} skipped_overlap={} skipped_label_unknown={} skipped_empty={}",
                len(selected_rows), args.n, skipped_overlap, skipped_label_unknown, skipped_empty)
    if len(selected_rows) < args.n:
        raise RuntimeError(
            f"Only {len(selected_rows)} papers passed all filters out of {args.max_fetch} scanned. "
            "Increase --max-fetch or lower --from-year.")

    holdout_ids = set(row["doc_id"] for row in selected_rows)
    intersect = holdout_ids & train_ids
    assert not intersect, f"BUG: {len(intersect)} holdout ids overlap training: {sorted(list(intersect))[:5]}"
    logger.info("Overlap assertion passed: holdout ∩ training = ∅")

    # Citations: keep only edges where source is a holdout paper and target is in
    # holdout ∪ training. OpenAlex only returns outgoing references, so the reverse
    # direction (training papers citing holdout) is not available and is skipped.
    valid_targets = holdout_ids | train_ids
    edge_rows: list[dict[str, int]] = []
    for row in selected_rows:
        src = int(row["doc_id"])
        for tgt in row["_referenced_works"]:
            if tgt in valid_targets and tgt != src:
                edge_rows.append({"source": src, "target": tgt})

    documents_payload = [
        {k: v for k, v in row.items() if k != "_referenced_works"} for row in selected_rows]
    documents = pl.DataFrame(
        documents_payload,
        schema={
            "doc_id": pl.Int64, "title": pl.String, "abstract": pl.String,
            "venue": pl.String, "publisher": pl.String,
            "authors": pl.List(pl.String), "year": pl.Int64, "label": pl.String})

    citations = pl.DataFrame(edge_rows, schema={"source": pl.Int64, "target": pl.Int64})

    documents.write_parquet(out_dir / "documents.parquet")
    citations.write_parquet(out_dir / "citations.parquet")
    logger.info("Wrote {} documents and {} citations to {}",
                documents.height, citations.height, out_dir)

    # Baselines: copy the original schema with zero rows so the eval pipeline can
    # still read the file unconditionally. Skip silently if the training baselines
    # file is absent (it is in some local checkouts).
    if args.train_baselines.exists():
        baseline_schema = pl.read_parquet(args.train_baselines).schema
        empty_baselines = pl.DataFrame(schema=baseline_schema)
        empty_baselines.write_parquet(out_dir / "baselines.parquet")
        logger.info("Wrote empty baselines.parquet with schema {} -> {}",
                    dict(baseline_schema), out_dir / "baselines.parquet")
    else:
        logger.warning("Training baselines not found at {}; skipping baselines.parquet emission.",
                       args.train_baselines)

    # Manifest with provenance — useful for cross-checking that the same holdout
    # was scored when comparing runs across machines.
    manifest = {
        "n_holdout": documents.height,
        "n_citations": citations.height,
        "train_docs_path": str(args.train_docs),
        "from_year": args.from_year,
        "max_fetch": args.max_fetch,
        "skipped_overlap": skipped_overlap,
        "skipped_label_unknown": skipped_label_unknown,
        "skipped_empty": skipped_empty,
        "holdout_doc_ids": sorted(int(v) for v in documents["doc_id"].to_list())}
    (out_dir / "holdout_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote manifest -> {}", out_dir / "holdout_manifest.json")


if __name__ == "__main__":
    main()
