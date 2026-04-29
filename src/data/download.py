"""
CLI entry point for building MetaGraphSci datasets from OpenAlex.

The script is intentionally thin: it parses arguments, validates cross-argument
constraints, prints a concise execution summary, then delegates all real work
to the downloader module. No graph or labeling logic lives here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from downloaders import (
    download_forc2025, download_ogbn_arxiv,
    download_openalex,  download_planetoid_dataset)


DEFAULT_CONFIG_TEMPLATE = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


"""Download and normalize supported benchmarks for MetaGraphSci.

- fetch supported benchmarks from their original sources,
- map heterogeneous fields onto the shared project schema,
- export normalized documents and citations tables,
- generate a matching config so experiments can start immediately.
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset download commands."""
    parser = argparse.ArgumentParser(description="Download datasets used by the MetaGraphSci analysis pipeline.")

    parser.add_argument("--dataset", required=True, choices=["cora", "pubmed", "ogbn_arxiv", "forc4cl", "openalex"], help="Target benchmark to process.")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output directory for normalized tables and configs.")
    parser.add_argument("--config_template", default=DEFAULT_CONFIG_TEMPLATE, type=Path, help="Base YAML config template to clone.")

    parser.add_argument("--oa_filter", default="primary_topic.field.id:17",
                        help=("OpenAlex /works filter expression. OpenAlex only filters topics by ID, not display name. "
                              "Examples: 'primary_topic.field.id:17' (Computer Science), "
                              "'primary_topic.subfield.id:1702' (Artificial Intelligence), "
                              "'primary_topic.id:T10320' (a specific topic). "
                              "Browse IDs at https://api.openalex.org/fields and /subfields."))
    parser.add_argument("--oa_max_works", default=50_000, type=int, help="Maximum number of OpenAlex works to fetch.")
    parser.add_argument("--oa_email", default="", help="Contact email for the OpenAlex polite pool (higher rate limits).")
    parser.add_argument("--oa_label_field", default="field", choices=["field", "subfield", "topic", "domain"],
                        help="Which level of the OpenAlex topic hierarchy to use as the classification label.")
    parser.add_argument("--oa_workers", default=1, type=int,
                        help="Number of parallel download workers. 1 (default) = serial. "
                             "Workers partition the [--oa_year_min, --oa_year_max] range into disjoint year buckets. "
                             "Recommended max ~4 to stay within the OpenAlex polite-pool rate limit.")
    parser.add_argument("--oa_year_min", default=2000, type=int,
                        help="Lower bound (inclusive) for year-based parallel partitioning. Ignored if --oa_workers=1.")
    parser.add_argument("--oa_year_max", default=None, type=int,
                        help="Upper bound (inclusive) for year-based parallel partitioning. Defaults to the current year. "
                             "Ignored if --oa_workers=1.")

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
    if args.dataset == "cora":
        download_planetoid_dataset("Cora", args.out_dir, args.config_template)
    elif args.dataset == "pubmed":
        download_planetoid_dataset("PubMed", args.out_dir, args.config_template)
    elif args.dataset == "ogbn_arxiv":
        download_ogbn_arxiv(args.out_dir, args.config_template)
    elif args.dataset == "forc4cl":
        download_forc2025(args.out_dir, args.config_template)
    elif args.dataset == "openalex":
        download_openalex(
            args.out_dir,
            args.config_template,
            filter_str=args.oa_filter,
            max_works=args.oa_max_works,
            email=args.oa_email,
            label_field=args.oa_label_field,
            workers=args.oa_workers,
            year_min=args.oa_year_min,
            year_max=args.oa_year_max,
        )


if __name__ == "__main__":
    main()
