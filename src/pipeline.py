from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import yaml

from data import (
    MultiScaleDocumentDataset,
    build_loader, build_neighbor_cache,
    create_encoders, create_low_label_split, create_tokenizer,
    load_citation_graph, load_documents, load_neighbor_cache,
    save_neighbor_cache, split_documents, split_graphs)
from eval import (
    PRIMARY_METRICS,
    aggregate_seed_results, plot_seed_metric_trend,
    plot_calibration, plot_embedding_projection, plot_training_history,
    precision_recall_fscore_support, plot_per_class_f1, plot_pseudo_label_ratio,
    save_benchmark_table, save_frame, save_evaluation_bundle)
from model import MetaGraphSci
from train import MetaGraphSciTrainer

"""Assemble end-to-end experiments for MetaGraphSci.

This file turns a config into a real experiment run. It owns split creation,
cache reuse, dataset construction, model instantiation, and trainer setup.

- keep experiment assembly explicit and easy to trace,
- build the citation-context cache only when needed,
- hand off ready-to-run datasets, models, and trainers.
"""


def read_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text()) or {}
    raise ValueError(f"Unsupported config format: {path.suffix}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def labeled_prior(documents: pd.DataFrame, num_classes: int) -> list[float]:
    """Determines the empirical class distribution of the supervised subset.
    
    This prior is used later by the pseudo-labeler to ensure the model doesn't 
    collapse and over-predict the majority class on unlabeled data.
    """
    counts = documents["label"].dropna().astype(int).value_counts().sort_index()
    prior = np.zeros(num_classes, dtype=np.float32)
    for label, count in counts.items():
        prior[label] = float(count)
    
    # Normalize into a valid probability distribution
    prior = prior / np.maximum(prior.sum(), 1.0)
    return prior.tolist()


def infer_max_authors(docs: pd.DataFrame, fallback: int = 8, cap: int = 32) -> int:
    """Calculates a robust upper bound for author sequence lengths.
    
    Instead of tracking the absolute maximum number of authors (which could be hundreds 
    for massive collaborative physics or medical papers), this caps the dimension at the 
    95th percentile to prevent out-of-memory errors during tensor allocation.
    """
    if "authors" not in docs.columns:
        return fallback

    counts = [len(values) if isinstance(values, list) else 0 for values in docs["authors"].tolist()]
    if not counts:
        return fallback
    return max(1, min(int(np.percentile(counts, 95)), cap))


def context_budget(data_cfg: dict[str, Any]) -> int:
    """Resolves the maximum number of neighbor documents to include in a node's context.
    
    Prioritizes explicit context limits if provided; otherwise, it deduces the limit 
    from the hop profile to keep memory usage predictable.
    """
    if "max_context_size" in data_cfg:
        return int(data_cfg["max_context_size"])
    if "max_neighbors_per_hop" in data_cfg:
        return int(sum(data_cfg["max_neighbors_per_hop"]))
    return 16


def build_dataset(
    docs: pd.DataFrame, context_docs: pd.DataFrame,
    tokenizer: Any, encoders: dict[str, dict[str, int]],
    context_cache: dict[int, list[dict[str, int | float]]],
    data_cfg: dict[str, Any]) -> MultiScaleDocumentDataset:
    """Binds raw documents, metadata encoders, and retrieved graph contexts into a unified dataset ready for the PyTorch DataLoader."""
    max_authors = data_cfg.get("max_authors")
    if max_authors is None:
        max_authors = infer_max_authors(docs)

    return MultiScaleDocumentDataset(
        docs, tokenizer, encoders["venue"], encoders["publisher"], encoders["author"],
        data_cfg["max_seq_length"], context_budget(data_cfg), max_authors, context_documents=context_docs, context_cache=context_cache,
        cache_text=data_cfg.get("cache_text", True), pretokenize_context=data_cfg.get("pretokenize_context", False),
        hop_profile_dim=data_cfg.get("k_hops", 2), spectral_dim=data_cfg.get("spectral_dim", 0))


def cache_metadata(data_cfg: dict[str, Any], node_ids: list[int], seed: int) -> dict[str, Any]:
    """Generates a structural fingerprint of the data pipeline settings.
    
    This metadata is saved alongside the citation context cache to verify that 
    the cache was generated under the exact same topological and sampling rules.
    """
    return {"max_context_size": context_budget(data_cfg), 
            "sampling_strategy": data_cfg.get("sampling_strategy", "local_relevance"),
            "num_nodes": len(node_ids), "seed": seed,
            "k_hops": int(data_cfg.get("k_hops", 2)),
            "spectral_dim": int(data_cfg.get("spectral_dim", 0)),
            "enable_spectral": bool(data_cfg.get("enable_spectral", False))}


def cache_is_compatible(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Validates if an existing on-disk cache matches the current experiment's structural fingerprint to safely bypass graph sampling."""
    return all(metadata.get(key) == expected.get(key) for key in ("max_context_size", "sampling_strategy", "num_nodes", "k_hops", "spectral_dim", "enable_spectral"))


def neighbor_sets(context_cache: dict[int, list[dict[str, int | float]]]) -> dict[int, set[int]]:
    """Extracts flat sets of neighboring document IDs from the detailed context cache for rapid inclusion/exclusion checks during contrastive masking."""
    return {node_id: {int(entry["doc_id"]) for entry in entries if int(entry.get("doc_id", 0)) > 0} for node_id, entries in context_cache.items()}


def build_run_bundle(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Orchestrates the data pipeline for a single experiment seed.
    
    This function handles the heavy lifting before training begins:
    1. Splits the document corpus into train, val, and test subsets.
    2. Builds categorical encoders mapping text metadata to model embedding indices.
    3. Manages the expensive retrieval of local neighborhood subgraphs, intelligently loading from disk if compatible rules apply.
    4. Packages everything into datasets tailored for self-supervised pretraining, fine-tuning, and evaluation.
    """
    data_cfg, project_cfg = cfg["data"], cfg["project"]

    # Load core topologies
    docs, label_names = load_documents(data_cfg["documents"], label_column=data_cfg["label_column"])
    graph = load_citation_graph(data_cfg["citations"], source_col=data_cfg["source_col"], target_col=data_cfg["target_col"], node_ids=docs["doc_id"].tolist())

    # Form isolated splits to prevent target leakage during neighborhood sampling
    train_docs, val_docs, test_docs = split_documents(docs, test_size=data_cfg["test_size"], val_size=data_cfg["val_size"], seed=seed, strategy=data_cfg["split_strategy"])
    labeled_docs, unlabeled_docs = create_low_label_split(train_docs, label_ratio=data_cfg["label_ratio"], seed=seed)
    graphs = split_graphs(graph, train_docs["doc_id"].tolist(), val_docs["doc_id"].tolist(), test_docs["doc_id"].tolist(), mode=data_cfg["graph_mode"])

    # Prepare input mappings
    tokenizer = create_tokenizer(cfg["model"]["tokenizer_name"])
    encoders = {key: value for key, value in zip(["venue", "publisher", "author"], create_encoders(train_docs))}

    cache_root = Path(project_cfg["cache_dir"]) / project_cfg["benchmark"] / f"seed_{seed}"
    cache_root.mkdir(parents=True, exist_ok=True)

    def build_or_load_cache(split_graph, node_ids: list[int], cache_name: str, valid_ids: list[int] | None = None) -> dict[int, list[dict[str, int | float]]]:
        """Local helper to short-circuit expensive graph walks if a valid cache exists on disk."""
        cache_path = cache_root / f"{cache_name}_context_cache.json"
        expected_metadata = cache_metadata(data_cfg, node_ids=node_ids, seed=seed)

        if cache_path.exists():
            cache, metadata = load_neighbor_cache(cache_path)
            if cache_is_compatible(metadata, expected_metadata):
                return cache

        # Perform the actual graph traversal and sampling if no valid cache is found
        cache = build_neighbor_cache(
            split_graph, node_ids, train_docs, context_budget(data_cfg), valid_node_ids=valid_ids,
            sampling_strategy=data_cfg.get("sampling_strategy", "local_relevance"),
            connectivity_weight=data_cfg["connectivity_weight"],
            temporal_weight=data_cfg["temporal_weight"],
            reciprocity_weight=data_cfg["reciprocity_weight"],
            overlap_weight=data_cfg["overlap_weight"],
            k_hops=data_cfg.get("k_hops", 2),
            spectral_dim=data_cfg.get("spectral_dim", 0),
            enable_spectral=data_cfg.get("enable_spectral", False))
            
        save_neighbor_cache(cache, cache_path, expected_metadata)
        return cache

    # Resolve context dependencies across all dataset splits
    train_cache = build_or_load_cache(graphs["pretrain"], train_docs["doc_id"].tolist(), "train", valid_ids=train_docs["doc_id"].tolist())
    val_cache = build_or_load_cache(graphs["val"], val_docs["doc_id"].tolist(), "val", valid_ids=val_docs["doc_id"].tolist())
    test_cache = build_or_load_cache(graphs["test"], test_docs["doc_id"].tolist(), "test", valid_ids=test_docs["doc_id"].tolist())

    # Transductive environments allow the model to see future node features (but not labels), whereas inductive restricts entirely to the training set.
    common_context = docs if data_cfg["graph_mode"] == "transductive" else train_docs
    
    datasets = {
        "pretrain": build_dataset(train_docs, common_context, tokenizer, encoders, train_cache, data_cfg),
        "labeled": build_dataset(labeled_docs, common_context, tokenizer, encoders, train_cache, data_cfg),
        "unlabeled": build_dataset(unlabeled_docs, common_context, tokenizer, encoders, train_cache, data_cfg),
        "val": build_dataset(val_docs, docs if data_cfg["graph_mode"] == "transductive" else val_docs, tokenizer, encoders, val_cache, data_cfg),
        "test": build_dataset(test_docs, docs if data_cfg["graph_mode"] == "transductive" else test_docs, tokenizer, encoders, test_cache, data_cfg)}

    num_classes = len(label_names) if label_names else int(len(sorted(docs["label"].dropna().astype(int).unique().tolist())))
    
    return {
        "documents": docs, "graph": graph, "label_names": label_names, "encoders": encoders, "datasets": datasets,
        "train_neighbor_cache": neighbor_sets(train_cache), "num_classes": num_classes, "labeled_prior": labeled_prior(labeled_docs, num_classes)}


def build_model(bundle: dict[str, Any], cfg: dict[str, Any]) -> MetaGraphSci:
    """Constructs the MetaGraphSci architecture dynamically based on the dimensions extracted during data bundling."""
    model_cfg, data_cfg = cfg["model"], cfg["data"]
    return MetaGraphSci(
        num_classes=bundle["num_classes"],
        num_venues=max(bundle["encoders"]["venue"].values()) + 1,
        num_publishers=max(bundle["encoders"]["publisher"].values()) + 1,
        num_authors=max(bundle["encoders"]["author"].values()) + 1,
        text_dim=model_cfg["text_dim"], metadata_dim=model_cfg["metadata_dim"], citation_dim=model_cfg["citation_dim"], fusion_dim=model_cfg["fusion_dim"],
        classifier_scale=model_cfg["classifier_scale"], model_name=model_cfg["tokenizer_name"], ablation_mode="full",
        peft_mode=model_cfg["peft_mode"], lora_r=model_cfg["lora_r"], lora_alpha=model_cfg["lora_alpha"], lora_dropout=model_cfg["lora_dropout"],
        peft_target_modules=None, gradient_checkpointing=model_cfg["gradient_checkpointing"], freeze_backbone_until_layer=model_cfg["freeze_backbone_until_layer"],
        citation_heads=model_cfg["citation_heads"], citation_layers=model_cfg["citation_layers"], citation_ff_dim=model_cfg["citation_ff_dim"],
        selector_hidden_dim=model_cfg["selector_hidden_dim"], selector_top_k=model_cfg["selector_top_k"],
        max_context_size=context_budget(data_cfg), fusion_modality_dropout=model_cfg["fusion_modality_dropout"], citation_dropout=model_cfg["citation_dropout"],
        hop_profile_dim=data_cfg.get("k_hops", 2), spectral_dim=data_cfg.get("spectral_dim", 0))


def summarize_run(result: dict[str, Any]) -> dict[str, Any]:
    """Extracts primary top-level evaluation metrics from the hierarchical training result bundle for easy tabular logging."""
    summary = {"best_score": result["finetune"]["best_score"]}
    if result.get("test"):
        summary.update(result["test"]["metrics"])
    return summary


def parse_args() -> argparse.Namespace:
    """Parses command-line overrides, allowing temporary runtime adjustments without modifying the underlying config files."""
    parser = argparse.ArgumentParser(description="Run MetaGraphSci experiments from config.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--benchmark", default=None, help="Optional benchmark override")
    parser.add_argument("--documents", default=None, help="Optional documents override")
    parser.add_argument("--citations", default=None, help="Optional citations override")
    parser.add_argument("--baselines", default=None, help="Optional baselines override")
    return parser.parse_args()


def main() -> None:
    """Drives the execution of the entire experiment grid. 
    
    It iterates over requested architectural ablations and random seeds, spins up the 
    trainer for each configuration, and aggregates the final evaluation metrics into 
    consolidated benchmark reports (CSV, charts, and LaTeX tables).
    """
    args = parse_args()
    cfg = read_config(args.config)

    # Apply any CLI overrides directly to the configuration dictionary
    if args.benchmark:
        cfg.setdefault("project", {})["benchmark"] = args.benchmark
    if args.documents:
        cfg.setdefault("data", {})["documents"] = args.documents
    if args.citations:
        cfg.setdefault("data", {})["citations"] = args.citations
    if args.baselines:
        cfg.setdefault("data", {})["baselines"] = args.baselines

    benchmark, seeds, ablations = cfg["project"]["benchmark"], list(cfg["train"]["seeds"]), list(cfg["train"]["ablations"])
    output_root = Path(cfg["project"]["output_dir"]) / benchmark
    output_root.mkdir(parents=True, exist_ok=True)

    # Optional: load external baseline metrics to append to final reporting tables
    baseline_rows: list[dict[str, Any]] = []
    baselines_path = cfg["data"].get("baselines")
    if baselines_path and Path(baselines_path).exists():
        baseline_rows = pl.read_csv(baselines_path).to_dicts()

    all_run_rows: list[dict[str, Any]] = []
    
    # Outer loop defines the architecture type, inner loop evaluates stability across random splits
    for ablation in ablations:
        for seed in seeds:
            set_seed(seed)
            bundle = build_run_bundle(cfg, seed)
            run_dir = output_root / ablation / f"seed_{seed}"
            artifact_root = run_dir / "artifacts"
            artifact_root.mkdir(parents=True, exist_ok=True)

            # Bind hardware configurations to the trainer
            trainer_cfg = {
                "output_dir": str(run_dir), "mixed_precision": "fp16" if torch.cuda.is_available() else "no",
                "gradient_accumulation_steps": cfg["trainer"].get("gradient_accumulation_steps", 1),
                "ablation_mode": ablation, "run_name": f"{benchmark}_{ablation}_seed_{seed}", **cfg["trainer"]}

            trainer = MetaGraphSciTrainer(
                model=build_model(bundle, cfg), citation_graph=bundle["graph"], neighbor_cache=bundle["train_neighbor_cache"],
                config=trainer_cfg, label_names=bundle["label_names"], labeled_class_prior=bundle["labeled_prior"])

            train_cfg = cfg["train"]

            # Build loaders once so they can be reused for extra evaluation artifact generation
            pretrain_loader = build_loader(
                bundle["datasets"]["pretrain"], batch_size=train_cfg["batch_size"],
                shuffle=True, num_workers=train_cfg.get("num_workers", 0))

            labeled_loader = build_loader(
                bundle["datasets"]["labeled"], batch_size=train_cfg["batch_size"],
                shuffle=True, num_workers=train_cfg.get("num_workers", 0))

            unlabeled_loader = build_loader(
                bundle["datasets"]["unlabeled"], batch_size=train_cfg["batch_size"],
                shuffle=True, num_workers=train_cfg.get("num_workers", 0))

            val_loader = build_loader(
                bundle["datasets"]["val"], batch_size=train_cfg["batch_size"],
                shuffle=False, num_workers=train_cfg.get("num_workers", 0))

            test_loader = build_loader(
                bundle["datasets"]["test"], batch_size=train_cfg["batch_size"],
                shuffle=False, num_workers=train_cfg.get("num_workers", 0))

            # Execute the full pipeline: Pretraining -> Finetuning -> Testing
            result = trainer.train_full_pipeline(
                pretrain_loader=pretrain_loader,
                labeled_loader=labeled_loader,
                unlabeled_loader=unlabeled_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                pretrain_epochs=train_cfg["pretrain_epochs"], finetune_epochs=train_cfg["finetune_epochs"])

            # Save additional validation artifacts in the same style as the built-in test bundle
            val_result = trainer.evaluate(val_loader, split="val", returnembeddings=True)
            save_evaluation_bundle(
                val_result["bundle"], artifact_root / "val", "val",
                val_result["y_true"], val_result["y_pred"],
                y_prob=val_result["y_prob"], embeddings=val_result.get("embeddings"),
                label_names=bundle["label_names"], history_rows=result.get("history_rows", []))

            # Optional extra embedding projections for validation
            if val_result.get("embeddings") is not None:
                plot_embedding_projection(
                    val_result["embeddings"], val_result["y_true"],
                    artifact_root / "val" / "val_tsne.png",
                    label_names=bundle["label_names"], method="tsne")
                plot_embedding_projection(
                    val_result["embeddings"], val_result["y_true"],
                    artifact_root / "val" / "val_pca.png",
                    label_names=bundle["label_names"], method="pca")

            # Add extra embedding projections for the already saved test bundle
            if result.get("test") and result["test"].get("embeddings") is not None:
                plot_embedding_projection(
                    result["test"]["embeddings"], result["test"]["y_true"],
                    artifact_root / "test" / "test_tsne.png",
                    label_names=bundle["label_names"], method="tsne")
                plot_embedding_projection(
                    result["test"]["embeddings"], result["test"]["y_true"],
                    artifact_root / "test" / "test_pca.png",
                    label_names=bundle["label_names"], method="pca")

            # Capture key metrics from this specific run
            summary = summarize_run(result)
            (artifact_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
            all_run_rows.append({
                "method": "MetaGraphSci", "dataset": benchmark, "ablation": ablation, "seed": seed,
                **{metric: summary.get(metric, np.nan) for metric in PRIMARY_METRICS}})

    # Compute means and standard deviations across all seeds to determine statistical significance
    results_all, results_summary = aggregate_seed_results(all_run_rows, metrics=PRIMARY_METRICS)
    
    # Export artifacts to the benchmark directory
    save_frame(results_all, output_root / "results_all_runs.csv")
    save_frame(results_summary, output_root / "results_summary.csv")

    for metric in PRIMARY_METRICS:
        plot_seed_metric_trend(results_all, output_root / f"results_{metric}_seeds.png", metric=metric)

    # Combine with external baselines if available for the final LaTeX publication table
    comparison_rows = results_summary.to_dicts()
    if baseline_rows:
        comparison_rows.extend(baseline_rows)
    if comparison_rows:
        comparison_frame = pl.DataFrame(comparison_rows)
        save_frame(comparison_frame, output_root / "comparison_summary.csv")
        save_benchmark_table(comparison_frame, output_root / "comparison_summary.tex", datasets=[benchmark], metrics=PRIMARY_METRICS)

    save_benchmark_table(results_summary, output_root / "results_summary.tex", datasets=[benchmark], metrics=PRIMARY_METRICS)
    print(f"Saved run artifacts to {output_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Experiment failed with error: {e}")
