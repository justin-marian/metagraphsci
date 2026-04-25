"""
CLI entry point for building MetaGraphSci datasets from OpenAlex.

The script is intentionally thin: it parses arguments, validates cross-argument
constraints, prints a concise execution summary, then delegates all real work
to the downloader module. No graph or labeling logic lives here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from domain_configs import DOMAIN_REGISTRY
from downloaders import download_openalex_query


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments for dataset construction."""
    parser = argparse.ArgumentParser(description="Build a MetaGraphSci dataset from OpenAlex.", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    # --- Core options ---
    parser.add_argument("--domain", default="cs_ai", choices=list(DOMAIN_REGISTRY.keys()), help="Research domain to build. Default: cs_ai.")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output directory for documents.csv, citations.csv and config.yaml.")
    parser.add_argument("--config_template", default="configs/config.yaml", type=Path, help="Base YAML config patched with dataset-specific paths and settings.",)
    # --- Fetch options ---
    fetch = parser.add_argument_group("Fetch options")
    fetch.add_argument("--query", type=str, default=None, metavar="TEXT", help="Optional free-text prefix applied to every per-class sub-query.")
    fetch.add_argument("--max_papers", type=int, metavar="N", help="Total papers across all classes.",)
    fetch.add_argument("--papers_per_class", type=int, default=None, metavar="N", help="Hard per-class cap. Overrides max_papers // n_classes when set.")
    fetch.add_argument("--classes", nargs="+", default=None, metavar="CLASS", help="Subset of label classes to include for the selected domain.")
    fetch.add_argument("--from_year", type=int, default=None, metavar="YYYY", help="Earliest publication year, inclusive.")
    fetch.add_argument("--to_year", type=int, default=None, metavar="YYYY", help="Latest publication year, inclusive.")
    fetch.add_argument("--no_citations", action="store_true", default=False, help="Skip citation graph creation and emit an empty citations.csv.")
    fetch.add_argument("--no_expand", action="store_true", default=False, help=(
        "Disable the citation-network expansion pass. By default the builder fetches the most cross-referenced external papers to raise "
        "citation density (avg deg << 1 without it). Use --no_expand only for fast debug runs."))
    fetch.add_argument("--llm_fallback", action="store_true", default=False, help="Use the Claude API for papers with no structural label signal.")
    fetch.add_argument("--llm_api_key", type=str, default=None, metavar="KEY", help="Anthropic API key. Falls back to ANTHROPIC_API_KEY env var if omitted.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate cross-argument constraints before any download work begins.

    Raises ValueError with a descriptive message when:
        - --classes contains a class name not in the selected domain's taxonomy.
        - --from_year is strictly greater than --to_year.
    """
    config = DOMAIN_REGISTRY[args.domain]

    if args.classes:
        invalid = [label for label in args.classes if label not in config.labels]
        if invalid:
            raise ValueError(f"Unknown class(es) for domain {args.domain!r}: {invalid}\n Available: {config.labels}")

    if (args.from_year is not None and args.to_year is not None and args.from_year > args.to_year):
        raise ValueError(f"--from_year ({args.from_year}) must be ≤ --to_year ({args.to_year})")


def resolve_target_sizes(args: argparse.Namespace) -> tuple[list[str], int, int]:
    """
    Resolve the active class list and the effective per-class paper budget.

    Returns
    -------
    classes          – Ordered list of class names to fetch.
    papers_per_class – Papers to fetch per class (at most).
    total_target     – Expected total papers (papers_per_class x n_classes).
    """
    config = DOMAIN_REGISTRY[args.domain]
    classes = args.classes or config.labels
    papers_per_class = args.papers_per_class or max(1, args.max_papers // len(classes))
    return classes, papers_per_class, papers_per_class * len(classes)


def print_summary(
    args: argparse.Namespace, classes: list[str],
    papers_per_class: int, total_target: int) -> None:
    """Print a concise execution plan before the download starts."""
    config = DOMAIN_REGISTRY[args.domain]
    print("\nOpenAlex dataset download")
    print(f"  Domain          : {args.domain} ({len(config.labels)} classes total)")
    print(f"  Output dir      : {args.out_dir}")
    print(f"  Query prefix    : {args.query or '(none)'}")
    print(f"  Classes ({len(classes):2d})    : {', '.join(classes)}")
    print(f"  Papers/class    : {papers_per_class}")
    print(f"  Total target    : {total_target}")
    print(f"  Year window     : {args.from_year or 'any'} - {args.to_year or 'any'}")
    print(f"  Citations       : {'no' if args.no_citations else 'yes'}")
    print(f"  Cite expand     : {'no' if args.no_expand else 'yes'}")
    print(f"  LLM fallback    : {'yes' if args.llm_fallback else 'no'}")
    print()


def main() -> None:
    """Run the full CLI workflow: validate → summarise → delegate to downloader."""
    args = parse_args()
    validate_args(args)
    classes, papers_per_class, total_target = resolve_target_sizes(args)
    print_summary(args, classes, papers_per_class, total_target)

    download_openalex_query(
        out_dir=args.out_dir,
        config_template=args.config_template,
        query=args.query,
        max_papers=args.max_papers,
        from_year=args.from_year,
        to_year=args.to_year,
        include_citations=not args.no_citations,
        expand_citations=not args.no_expand,
        use_llm_fallback=args.llm_fallback,
        llm_api_key=args.llm_api_key,
        papers_per_class=args.papers_per_class,
        classes=args.classes,
        domain=args.domain)


if __name__ == "__main__":
    main()
