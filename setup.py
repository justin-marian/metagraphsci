from setuptools import setup, find_packages

setup(
    name="MetaGraphSci",
    description="MetaGraphSci: Meta-Informed Graph Transformer Self-Supervised Contrastive Neighbor Adaptation for Scientific Document Classification",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "accelerate",
        "bitsandbytes",
        "matplotlib",
        "mlflow",
        "networkx",
        "numpy",
        "ogb",
        "pandas",
        "peft",
        "polars",
        "PyYAML",
        "scikit-learn",
        "seaborn",
        "torch",
        "torch-geometric",
        "transformers",
        "umap-learn",
        "wandb",
        "deepspeed",
    ]
)
