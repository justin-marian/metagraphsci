import argparse
from pathlib import Path

from downloaders import download_forc2025, download_ogbn_arxiv, download_planetoid_dataset


"""Download and normalize supported benchmarks for MetaGraphSci.

- fetch supported benchmarks from their original sources,
- map heterogeneous fields onto the shared project schema,
- export normalized documents and citations tables,
- generate a matching config so experiments can start immediately.
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset download commands."""
    parser = argparse.ArgumentParser(description="Download datasets used by the MetaGraphSci analysis pipeline.")
    
    parser.add_argument("--dataset", required=True, choices=["cora", "pubmed", "ogbn_arxiv", "forc4cl"], help="Target benchmark to process.")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output directory for normalized tables and configs.")
    parser.add_argument("--config_template", default="../configs/config.yaml", type=Path, help="Base YAML config template to clone.")
    
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
