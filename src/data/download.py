"""
CLI entry point for building MetaGraphSci datasets from OpenAlex and supported benchmarks.

The script is intentionally thin: it parses arguments, validates cross-argument
constraints, prints a concise execution summary, then delegates all real work to
the downloader module. No graph or labeling logic lives here.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .downloaders import (
    download_forc2025, download_ogbn_arxiv, download_openalex,
    download_planetoid_dataset)

DEFAULT_CONFIG_TEMPLATE = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
SUPPORTED_DATASETS = ("cora", "pubmed", "ogbn_arxiv", "forc4cl", "openalex")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset download commands."""
    parser = argparse.ArgumentParser(
        description="Download datasets used by the MetaGraphSci analysis pipeline.")

    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS,
                        help="Target benchmark to process.")
    parser.add_argument("--out_dir", required=True, type=Path,
                        help="Output directory for normalized tables and configs.")
    parser.add_argument("--config_template", default=DEFAULT_CONFIG_TEMPLATE, type=Path,
                        help="Base YAML config template to clone.")

    # OpenAlex-specific options. Ignored for Planetoid, OGBN-Arxiv, and FoRC2025.
    parser.add_argument("--oa_filter", default="primary_topic.field.id:17",
                        help=("OpenAlex /works filter expression. OpenAlex filters topics by ID, not display name. "
                            "Examples: 'primary_topic.field.id:17' for Computer Science, "
                            "'primary_topic.subfield.id:1702' for Artificial Intelligence, "
                            "'primary_topic.id:T10320' for a specific topic."))
    parser.add_argument("--oa_max_works", default=50_000, type=int,
                        help="Maximum number of OpenAlex works to fetch.")
    parser.add_argument("--oa_email", default="",
                        help="Contact email for the OpenAlex polite pool.")
    parser.add_argument("--oa_label_field", default="field",
                        choices=["field", "subfield", "topic", "domain"],
                        help="OpenAlex topic hierarchy level used as the classification label.")
    parser.add_argument("--oa_workers", default=1, type=int,
                        help=("Number of parallel OpenAlex workers. 1 = serial. Workers partition the "
                            "[--oa_year_min, --oa_year_max] range into disjoint year buckets."))
    parser.add_argument("--oa_year_min", default=2000, type=int,
                        help="Lower inclusive year bound for OpenAlex parallel partitioning.")
    parser.add_argument("--oa_year_max", default=None, type=int,
                        help="Upper inclusive year bound for OpenAlex partitioning. Defaults to current year.")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate cross-argument constraints before any download work begins."""
    if args.oa_max_works <= 0:
        raise ValueError("--oa_max_works must be positive")

    if args.oa_workers <= 0:
        raise ValueError("--oa_workers must be positive")

    if args.oa_year_max is not None and args.oa_year_min > args.oa_year_max:
        raise ValueError(f"--oa_year_min ({args.oa_year_min}) must be <= --oa_year_max ({args.oa_year_max})")

    if args.dataset != "openalex":
        return

    if not args.oa_filter.strip():
        raise ValueError("--oa_filter cannot be empty when --dataset=openalex")


def print_summary(args: argparse.Namespace) -> None:
    """Print a concise execution plan before the download starts."""
    print("\nMetaGraphSci dataset download")
    print(f"  Dataset         : {args.dataset}")
    print(f"  Output dir      : {args.out_dir}")
    print(f"  Config template : {args.config_template}")

    if args.dataset == "openalex":
        year_max = args.oa_year_max if args.oa_year_max is not None else datetime.now().year
        print(f"  OpenAlex filter : {args.oa_filter}")
        print(f"  Max works       : {args.oa_max_works}")
        print(f"  Label field     : {args.oa_label_field}")
        print(f"  Workers         : {args.oa_workers}")
        print(f"  Year window     : {args.oa_year_min} - {year_max}")
        print(f"  Polite email    : {args.oa_email or '(none)'}")

    print()


def download_dataset(args: argparse.Namespace) -> None:
    """Dispatch the selected dataset to its downloader."""
    if args.dataset == "cora":
        download_planetoid_dataset("Cora", args.out_dir, args.config_template)
        return

    if args.dataset == "pubmed":
        download_planetoid_dataset("PubMed", args.out_dir, args.config_template)
        return

    if args.dataset == "ogbn_arxiv":
        download_ogbn_arxiv(args.out_dir, args.config_template)
        return

    if args.dataset == "forc4cl":
        download_forc2025(args.out_dir, args.config_template)
        return

    if args.dataset == "openalex":
        download_openalex(
            args.out_dir, args.config_template,
            filter_str=args.oa_filter, max_works=args.oa_max_works,
            email=args.oa_email, label_field=args.oa_label_field,
            workers=args.oa_workers, year_min=args.oa_year_min,
            year_max=args.oa_year_max)
        return

    raise ValueError(f"Unsupported dataset: {args.dataset!r}")


def main() -> None:
    """Run the full CLI workflow: parse, validate, summarise, and delegate."""
    args = parse_args()
    validate_args(args)
    print_summary(args)
    download_dataset(args)


if __name__ == "__main__":
    main()
