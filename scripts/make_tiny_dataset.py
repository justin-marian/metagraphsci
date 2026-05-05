"""Generate a tiny subset of OpenAlex AI for fast smoke tests.

Usage:
    python scripts/make_tiny_dataset.py [N_DOCS]

Default N_DOCS=3000. Reads from data/openalex_ai/, writes to data/openalex_ai_tiny/.
Filters citations to keep only edges where both endpoints survived the truncation.
Filters baselines to surviving doc_ids when possible; otherwise copies as-is.
"""
import shutil
import sys
from pathlib import Path

import polars as pl

SRC = Path("data/openalex_ai")
DST = Path("data/openalex_ai_tiny")


def main(n_docs: int) -> None:
    DST.mkdir(parents=True, exist_ok=True)

    docs = pl.read_parquet(SRC / "documents.parquet").head(n_docs)
    surviving_ids = set(docs["doc_id"].to_list())
    docs.write_parquet(DST / "documents.parquet")

    citations = pl.read_parquet(SRC / "citations.parquet").filter(
        pl.col("source").is_in(surviving_ids) & pl.col("target").is_in(surviving_ids))
    citations.write_parquet(DST / "citations.parquet")

    baselines_path = SRC / "baselines.parquet"
    baselines = pl.read_parquet(baselines_path)
    if "doc_id" in baselines.columns:
        baselines = baselines.filter(pl.col("doc_id").is_in(surviving_ids))
        baselines.write_parquet(DST / "baselines.parquet")
    else:
        shutil.copy(baselines_path, DST / "baselines.parquet")

    print(f"Wrote {DST}/")
    print(f"  documents:  {len(docs):>7}")
    print(f"  citations:  {len(citations):>7}")
    print(f"  baselines:  {len(baselines):>7}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    main(n)
