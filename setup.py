from __future__ import annotations

from pathlib import Path
from setuptools import find_packages, setup


def get_long_description() -> str:
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.is_file():
        return readme_path.read_text(encoding="utf-8")
    return ""


def get_requirements() -> list[str]:
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.is_file():
        return []
    with req_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="MetaGraphSci",
    version="0.0.1",  # alpha inception
    author="Justin-Marian Popescu, Alexandra Chirita, Sirboiu Filip",
    description=(
        "MetaGraphSci is a semi-supervised model for scientific document classification combining metadata "
        "and citation graphs with contrastive learning and pseudo-labeling for low supervised settings."
    ),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/justin-marian/metagraphsci",
    project_urls={
        "Source Code": "https://github.com/justin-marian/metagraphsci",
    },
    keywords=[
        "gnn",
        "graph-transformer", 
        "self-supervised-learning",
        "document-classification",
        "pseudo-labeling",
        "pytorch",
    ],
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=get_requirements(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
