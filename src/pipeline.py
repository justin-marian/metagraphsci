import argparse
import json
import random
from pathlib import Path
from typing import Any, cast
import numpy as np
import pandas as pd
import polars as pl
import torch
import yaml
from loguru import logger

from .data import (
    MultiScaleDocumentDataset, NeighborCache, build_loader, build_neighbor_cache,
    create_encoders, create_low_label_split, create_tokenizer,
    load_citation_graph, load_documents, load_neighbor_cache,
    save_neighbor_cache, split_documents, split_graphs)
from .model.metagraphsci import MetaGraphSci
from train_eval import MetaGraphSciTrainerEval
from .include import (
    setup_global_logger,
    PRIMARY_METRICS, aggregate_seed_results, plot_embedding_projection,
    plot_seed_metric_trend, save_benchmark_table, save_evaluation_bundle, save_frame)


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
    """Determines the empirical class distribution of the supervised subset."""
    counts = documents["label"].dropna().astype(int).value_counts().sort_index()
    prior = np.zeros(num_classes, dtype=np.float32)
    for label, count in counts.items(): 
        prior[label] = float(count)
    return (prior / np.maximum(prior.sum(), 1.0)).tolist()


def infer_max_authors(docs: pd.DataFrame, fallback: int = 8, cap: int = 32) -> int:
    """Calculates a robust upper bound for author sequence lengths."""
    if "authors" not in docs.columns: 
        return fallback
    counts = [len(v) if isinstance(v, list) else 0 for v in docs["authors"].tolist()]
    return max(1, min(int(np.percentile(counts, 95)), cap)) if counts else fallback


def context_budget(data_cfg: dict[str, Any]) -> int:
    if "max_context_size" in data_cfg:
        return int(data_cfg["max_context_size"])
    if "max_neighbors_per_hop" in data_cfg: 
        return int(sum(data_cfg["max_neighbors_per_hop"]))
    return 16


def build_dataset(
    docs: pd.DataFrame, context_docs: pd.DataFrame, tokenizer: Any, encoders: dict[str, dict[str, int]],
    context_cache: NeighborCache, data_cfg: dict[str, Any]
) -> MultiScaleDocumentDataset:
    """Binds raw documents, encoders, and retrieved graph contexts into a PyTorch Dataset."""
    max_authors = data_cfg["max_authors"] or infer_max_authors(docs)
    return MultiScaleDocumentDataset(
        docs, tokenizer, encoders["venue"], encoders["publisher"], encoders["author"],
        data_cfg["max_seq_length"], context_budget(data_cfg), max_authors, context_documents=context_docs, 
        context_cache=context_cache, cache_text=data_cfg["cache_text"], 
        pretokenize_context=data_cfg["pretokenize_context"], hop_profile_dim=data_cfg["k_hops"], 
        spectral_dim=data_cfg["spectral_dim"])


def cache_metadata(data_cfg: dict[str, Any], node_ids: list[int], seed: int) -> dict[str, Any]:
    """Generates a structural fingerprint of the data pipeline settings."""
    return {
        "max_context_size": context_budget(data_cfg), "sampling_strategy": data_cfg["sampling_strategy"],
        "num_nodes": len(node_ids), "seed": seed, "k_hops": int(data_cfg["k_hops"]),
        "spectral_dim": int(data_cfg["spectral_dim"]), "enable_spectral": bool(data_cfg["enable_spectral"])
    }


def cache_is_compatible(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Validates if an existing on-disk cache safely matches current pipeline settings."""
    return all(metadata[k] == expected[k] for k in ("max_context_size", "sampling_strategy", "num_nodes", "k_hops", "spectral_dim", "enable_spectral"))


def neighbor_sets(context_cache: NeighborCache) -> dict[int, set[int]]:
    """Convert a serialized neighbor cache into a fast lookup mapping of node-to-neighbor IDs."""
    return {node_id: { int(cast(int, e["doc_id"])) for e in entries if int(cast(int, e["doc_id"])) > 0} for node_id, entries in context_cache.items()}


def build_run_bundle(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Orchestrates the data pipeline for a single experiment seed."""
    data_cfg, project_cfg = cfg["data"], cfg["project"]

    docs, label_names = load_documents(data_cfg["documents"], label_column=data_cfg["label_column"])
    graph = load_citation_graph(data_cfg["citations"], source_col=data_cfg["source_col"], target_col=data_cfg["target_col"], node_ids=docs["doc_id"].tolist())

    train_docs, val_docs, test_docs = split_documents(docs, test_size=data_cfg["test_size"], val_size=data_cfg["val_size"], seed=seed, strategy=data_cfg["split_strategy"])
    labeled_docs, unlabeled_docs = create_low_label_split(train_docs, label_ratio=data_cfg["label_ratio"], seed=seed)
    graphs = split_graphs(graph, train_docs["doc_id"].tolist(), val_docs["doc_id"].tolist(), test_docs["doc_id"].tolist(), mode=data_cfg["graph_mode"])

    tokenizer = create_tokenizer(cfg["model"]["tokenizer_name"])
    encoders = {key: value for key, value in zip(["venue", "publisher", "author"], create_encoders(train_docs))}

    cache_root = Path(project_cfg["cache_dir"]) / project_cfg["benchmark"] / f"seed_{seed}"
    cache_root.mkdir(parents=True, exist_ok=True)

    def build_or_load_cache(split_graph: Any, node_ids: list[int], cache_name: str, valid_ids: list[int] | None = None) -> NeighborCache:
        cache_path = cache_root / f"{cache_name}_context_cache.json"
        expected_meta = cache_metadata(data_cfg, node_ids=node_ids, seed=seed)

        if cache_path.exists():
            cache, meta = load_neighbor_cache(cache_path)
            if cache_is_compatible(meta, expected_meta): 
                return cache

        cache = build_neighbor_cache(
            split_graph, node_ids, train_docs, context_budget(data_cfg), valid_node_ids=valid_ids,
            sampling_strategy=data_cfg["sampling_strategy"],
            connectivity_weight=data_cfg["connectivity_weight"], temporal_weight=data_cfg["temporal_weight"],
            reciprocity_weight=data_cfg["reciprocity_weight"], overlap_weight=data_cfg["overlap_weight"],
            k_hops=data_cfg["k_hops"], spectral_dim=data_cfg["spectral_dim"], enable_spectral=data_cfg["enable_spectral"])
            
        save_neighbor_cache(cache, cache_path, expected_meta)
        return cache

    train_cache = build_or_load_cache(graphs["pretrain"], train_docs["doc_id"].tolist(), "train", valid_ids=train_docs["doc_id"].tolist())
    val_cache = build_or_load_cache(graphs["val"], val_docs["doc_id"].tolist(), "val", valid_ids=val_docs["doc_id"].tolist())
    test_cache = build_or_load_cache(graphs["test"], test_docs["doc_id"].tolist(), "test", valid_ids=test_docs["doc_id"].tolist())

    common_context = docs if data_cfg["graph_mode"] == "transductive" else train_docs
    datasets = {
        "pretrain": build_dataset(train_docs, common_context, tokenizer, encoders, train_cache, data_cfg),
        "labeled": build_dataset(labeled_docs, common_context, tokenizer, encoders, train_cache, data_cfg),
        "unlabeled": build_dataset(unlabeled_docs, common_context, tokenizer, encoders, train_cache, data_cfg),
        "val": build_dataset(val_docs, docs if data_cfg["graph_mode"] == "transductive" else val_docs, tokenizer, encoders, val_cache, data_cfg),
        "test": build_dataset(test_docs, docs if data_cfg["graph_mode"] == "transductive" else test_docs, tokenizer, encoders, test_cache, data_cfg)
    }

    num_classes = len(label_names) if label_names else int(len(sorted(docs["label"].dropna().astype(int).unique().tolist())))
    return {
        "documents": docs, "graph": graph, "label_names": label_names, "encoders": encoders, "datasets": datasets,
        "train_neighbor_cache": neighbor_sets(train_cache), "num_classes": num_classes, "labeled_prior": labeled_prior(labeled_docs, num_classes)
    }


def build_model(bundle: dict[str, Any], cfg: dict[str, Any]) -> MetaGraphSci:
    model_cfg, data_cfg = cfg["model"], cfg["data"]
    return MetaGraphSci(
        num_classes=bundle["num_classes"], num_venues=max(bundle["encoders"]["venue"].values()) + 1,
        num_publishers=max(bundle["encoders"]["publisher"].values()) + 1, num_authors=max(bundle["encoders"]["author"].values()) + 1,
        
        text_dim=model_cfg["text_dim"], metadata_dim=model_cfg["metadata_dim"], 
        citation_dim=model_cfg["citation_dim"], fusion_dim=model_cfg["fusion_dim"],
        classifier_scale=model_cfg["classifier_scale"], model_name=model_cfg["tokenizer_name"], ablation_mode="full",
        
        metadata_embedding_dim=model_cfg["metadata_embedding_dim"],
        metadata_cross_layers=model_cfg["metadata_cross_layers"],

        peft_mode=model_cfg["peft_mode"], lora_r=model_cfg["lora_r"], lora_alpha=model_cfg["lora_alpha"], lora_dropout=model_cfg["lora_dropout"],
        peft_target_modules=None, gradient_checkpointing=model_cfg["gradient_checkpointing"], freeze_backbone_until_layer=model_cfg["freeze_backbone_until_layer"],
        
        citation_heads=model_cfg["citation_heads"], citation_layers=model_cfg["citation_layers"], citation_ff_dim=model_cfg["citation_ff_dim"],
        selector_hidden_dim=model_cfg["selector_hidden_dim"], selector_top_k=model_cfg["selector_top_k"], max_context_size=context_budget(data_cfg), 
        fusion_modality_dropout=model_cfg["fusion_modality_dropout"], citation_dropout=model_cfg["citation_dropout"],
        
        hop_profile_dim=data_cfg["k_hops"], spectral_dim=data_cfg["spectral_dim"],
        use_latent_graph=model_cfg["use_latent_graph"], latent_graph_top_k=model_cfg["latent_graph_top_k"],
        hybrid_alpha_init=model_cfg["hybrid_alpha_init"])


def summarize_run(result: dict[str, Any]) -> dict[str, Any]:
    summary = {"best_score": result["finetune"]["best_score"]}
    summary.update(result["test"]["metrics"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MetaGraphSci experiments from config.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--benchmark", default=None, help="Optional benchmark override")
    parser.add_argument("--documents", default=None, help="Optional documents override")
    parser.add_argument("--citations", default=None, help="Optional citations override")
    parser.add_argument("--baselines", default=None, help="Optional baselines override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)

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

    baseline_rows: list[dict[str, Any]] = []
    baselines_path = cfg["data"]["baselines"]
    if baselines_path and Path(baselines_path).exists():
        baseline_rows = pl.read_csv(baselines_path).to_dicts()

    all_run_rows: list[dict[str, Any]] = []
    
    for ablation in ablations:
        for seed in seeds:
            set_seed(seed)
            bundle = build_run_bundle(cfg, seed)
            run_dir = output_root / ablation / f"seed_{seed}"
            artifact_root = run_dir / "artifacts"
            artifact_root.mkdir(parents=True, exist_ok=True)

            trainer_cfg = {
                "output_dir": str(run_dir), "mixed_precision": "fp16" if torch.cuda.is_available() else "no",
                "gradient_accumulation_steps": cfg["trainer"]["gradient_accumulation_steps"],
                "ablation_mode": ablation, "run_name": f"{benchmark}_{ablation}_seed_{seed}", **cfg["trainer"]
            }

            trainer = MetaGraphSciTrainerEval(
                model=build_model(bundle, cfg), citation_graph=bundle["graph"], neighbor_cache=bundle["train_neighbor_cache"],
                config=trainer_cfg, label_names=bundle["label_names"], labeled_class_prior=bundle["labeled_prior"])

            train_cfg = cfg["train"]
            pretrain_loader = build_loader(bundle["datasets"]["pretrain"], batch_size=train_cfg["batch_size"], shuffle=True, num_workers=train_cfg["num_workers"])
            labeled_loader = build_loader(bundle["datasets"]["labeled"], batch_size=train_cfg["batch_size"], shuffle=True, num_workers=train_cfg["num_workers"])
            unlabeled_loader = build_loader(bundle["datasets"]["unlabeled"], batch_size=train_cfg["batch_size"], shuffle=True, num_workers=train_cfg["num_workers"])
            val_loader = build_loader(bundle["datasets"]["val"], batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"])
            test_loader = build_loader(bundle["datasets"]["test"], batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"])

            result = cast(dict[str, Any], trainer.train_full_pipeline(
                pretrain_loader=pretrain_loader, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
                val_loader=val_loader, test_loader=test_loader,
                pretrain_epochs=train_cfg["pretrain_epochs"], finetune_epochs=train_cfg["finetune_epochs"]))

            val_result = trainer.evaluate(val_loader, split="val", return_embeddings=True)
            save_evaluation_bundle(
                val_result["bundle"], artifact_root / "val", "val", val_result["y_true"], val_result["y_pred"],
                y_prob=val_result["y_prob"], embeddings=val_result["embeddings"],
                label_names=bundle["label_names"], history_rows=result["history_rows"])

            plot_embedding_projection(val_result["embeddings"], val_result["y_true"], artifact_root / "val" / "val_tsne.png", label_names=bundle["label_names"], method="tsne")
            plot_embedding_projection(val_result["embeddings"], val_result["y_true"], artifact_root / "val" / "val_pca.png", label_names=bundle["label_names"], method="pca")

            plot_embedding_projection(result["test"]["embeddings"], result["test"]["y_true"], artifact_root / "test" / "test_tsne.png", label_names=bundle["label_names"], method="tsne")
            plot_embedding_projection(result["test"]["embeddings"], result["test"]["y_true"], artifact_root / "test" / "test_pca.png", label_names=bundle["label_names"], method="pca")

            summary = summarize_run(result)
            (artifact_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
            row_metrics = {metric: summary[metric] for metric in PRIMARY_METRICS if metric in summary}
            all_run_rows.append({"method": "MetaGraphSci", "dataset": benchmark, "ablation": ablation, "seed": seed, **row_metrics })

    results_all, results_summary = aggregate_seed_results(all_run_rows, metrics=PRIMARY_METRICS)
    save_frame(results_all, output_root / "results_all_runs.csv")
    save_frame(results_summary, output_root / "results_summary.csv")

    for metric in PRIMARY_METRICS: 
        plot_seed_metric_trend(results_all, output_root / f"results_{metric}_seeds.png", metric=metric)

    comparison_rows = results_summary.to_dicts() + baseline_rows
    if comparison_rows:
        comparison_frame = pl.DataFrame(comparison_rows)
        save_frame(comparison_frame, output_root / "comparison_summary.csv")
        save_benchmark_table(comparison_frame, output_root / "comparison_summary.tex", datasets=[benchmark], metrics=PRIMARY_METRICS)

    save_benchmark_table(results_summary, output_root / "results_summary.tex", datasets=[benchmark], metrics=PRIMARY_METRICS)
    logger.info(f"Saved run artifacts to {output_root}")


if __name__ == "__main__":
    try:
        setup_global_logger("..")
        main()
    except Exception as e:
        logger.info(f"Experiment failed with error: {e}")
