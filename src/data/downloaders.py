"""
Avoids importing the heavier OpenAlex implementation at module load time.
This file only provides the stable public entry point so external callers have a consistent
import path regardless of internal refactoring.
"""

from __future__ import annotations

from pathlib import Path


def download_openalex_query(
    out_dir:           str,
    config_template:   str | Path,
    query:             str | None = None,
    max_papers:        int = 5_000,
    from_year:         int | None = None,
    to_year:           int | None = None,
    include_citations: bool = True,
    expand_citations:  bool = True,
    use_llm_fallback:  bool = False,
    llm_api_key:       str | None = None,
    papers_per_class:  int | None = None,
    classes:           list[str] | None = None,
    domain:            str = "cs_ai") -> None:
    """
    Delegate OpenAlex dataset construction to the paper_labeler implementation.

    The local import keeps optional dependencies (polars, requests, etc.) and
    their startup cost out of code paths that only need the stable public interface.
    """
    from paper_labeler import download_openalex_query as impl
    impl(
        out_dir=out_dir, config_template=config_template,
        query=query, max_papers=max_papers,
        from_year=from_year, to_year=to_year,
        include_citations=include_citations, expand_citations=expand_citations,
        use_llm_fallback=use_llm_fallback, llm_api_key=llm_api_key,
        papers_per_class=papers_per_class, classes=classes, domain=domain)
