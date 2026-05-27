"""Inference-only evaluation of a trained MetaGraphSci checkpoint on a holdout slice.

Loads the best checkpoint from a completed training run, reuses the cached
encoders / tokenization / class space exactly as they were at training time,
and scores a held-out set of papers built by ``scripts/build_eval_holdout.py``.

Bypasses ``build_run_bundle`` deliberately — that helper re-splits the data and
re-fits the encoders on whatever it receives, which would corrupt the model
dimensions and the label-id mapping. Here every shared artifact is loaded from
its on-disk cache and the holdout becomes its own single dataset/loader.

Usage (run from repo root):
    python scripts/eval_holdout.py \\
        --config configs/openalex_ai_rtx5090_mid_1seed.yaml \\
        --holdout-dir data/openalex_ai_holdout100 \\
        --ablation full --seed 42 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference-only evaluation on a holdout slice.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the training YAML config.")
    parser.add_argument("--holdout-dir", type=Path, required=True,
                        help="Directory containing the holdout documents.parquet / citations.parquet.")
    parser.add_argument("--train-dir", type=Path, default=Path("data/openalex_ai"),
                        help="Original training dataset directory (provides label-id mapping).")
    parser.add_argument("--ablation", type=str, default="full",
                        help="Ablation name used at training (controls checkpoint subfolder).")
    parser.add_argument("--seed", type=int, default=42, help="Training seed (controls cache + run subfolder).")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Device to run inference on. Falls back to cpu automatically when cuda is missing.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-name", type=str, default="holdout100_eval",
                        help="Output stem written under the training run directory.")
    return parser.parse_args()


def force_device_env(device: str) -> str:
    """Pin the device BEFORE importing torch/accelerate so Accelerator picks it up.

    Accelerator() picks up CUDA the moment it is constructed inside
    MetaGraphSciTrainerEval.__init__, with no public API to override after the
    fact. Setting CUDA_VISIBLE_DEVICES="" up front is the cleanest way to force
    CPU; otherwise we leave the default GPU path.
    """
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"

    # Lazy-imported here so the cpu path above can keep CUDA hidden.
    import torch
    if not torch.cuda.is_available():
        # Don't lie about the device — fall back loudly so the user knows.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    return "cuda"


def normalise_holdout(
    raw_holdout: "pl.DataFrame", label_names: list[str], label_column: str = "label",
) -> "pl.DataFrame":
    """Normalise the holdout frame and apply the training label-id mapping.

    ``prepare_documents`` would otherwise resort the holdout labels alphabetically
    over a 100-row subset, producing a different id space than training. We
    therefore drop ``label`` before normalisation and re-attach it as a numeric
    column using the training ``label_names`` ordering.
    """
    import polars as pl  # noqa: F401  (forwarded into closure scope)
    from src.data.tabular_utils import prepare_documents

    if label_column not in raw_holdout.columns:
        raise ValueError(f"Holdout missing required '{label_column}' column.")

    raw_label_strings = [str(v) if v is not None else None
                         for v in raw_holdout[label_column].cast(pl.String).to_list()]
    mapping = {name: idx for idx, name in enumerate(label_names)}
    unknown = sorted(set(s for s in raw_label_strings if s is not None) - set(label_names))
    if unknown:
        # Should not happen — build_eval_holdout filters to known labels — but
        # surface the surprise loudly rather than silently producing -1 ids.
        raise ValueError(
            f"Holdout contains {len(unknown)} labels missing from training: {unknown[:5]}")

    # Strip the label column before prepare_documents so it doesn't re-derive
    # the id mapping on the 100-row slice.
    stripped = raw_holdout.drop(label_column)
    normalised, _ = prepare_documents(stripped, label_column=label_column)
    label_ids = [mapping[s] if s is not None else None for s in raw_label_strings]
    return normalised.with_columns(pl.Series(name="label", values=label_ids, dtype=pl.Int64))


def main() -> None:
    args = parse_args()
    device = force_device_env(args.device)

    from loguru import logger
    import polars as pl
    import pandas as pd
    import torch

    from src.data import (
        build_loader, build_neighbor_cache, build_tokenization_cache,
        create_tokenizer, load_citation_graph, load_documents,
        load_encoder_cache, load_tokenization_cache)
    from src.pipeline import (
        build_dataset, build_model, context_budget, infer_max_authors,
        labeled_prior, read_config)
    from src.train_eval import MetaGraphSciTrainerEval

    holdout_docs_path = args.holdout_dir / "documents.parquet"
    holdout_citations_path = args.holdout_dir / "citations.parquet"
    train_docs_path = args.train_dir / "documents.parquet"
    train_citations_path = args.train_dir / "citations.parquet"

    for path in (args.config, holdout_docs_path, holdout_citations_path,
                 train_docs_path, train_citations_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    cfg = read_config(args.config)
    project_cfg = cfg["project"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    trainer_cfg_in = dict(cfg.get("trainer", {}))

    # Override data paths in-memory; do not touch the YAML on disk.
    data_cfg["documents"] = str(holdout_docs_path)
    data_cfg["citations"] = str(holdout_citations_path)
    data_cfg["baselines"] = str(args.holdout_dir / "baselines.parquet")

    benchmark = project_cfg["benchmark"]
    run_dir = REPO_ROOT / project_cfg["output_dir"] / benchmark / args.ablation / f"seed_{args.seed}"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint missing at {checkpoint_path}. "
            "Confirm --ablation/--seed match the trained run and that runs/ exists on this host.")

    encoder_path = REPO_ROOT / project_cfg["cache_dir"] / f"seed_{args.seed}" / "encoders.json"
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Encoder cache missing at {encoder_path}. "
            "Inference requires the exact encoder vocab used at training time so the embedding shapes match.")

    logger.info("Loading training docs ({}) for label-id mapping and context", train_docs_path)
    train_docs, label_names = load_documents(str(train_docs_path), label_column=data_cfg["label_column"])
    if not label_names:
        raise RuntimeError("Training docs produced no label_names; cannot align holdout label ids.")
    num_classes = len(label_names)
    logger.info("Resolved {} classes from training label set", num_classes)

    logger.info("Loading holdout docs ({})", holdout_docs_path)
    raw_holdout = pl.read_parquet(holdout_docs_path)
    holdout_docs = normalise_holdout(raw_holdout, label_names, label_column=data_cfg["label_column"])

    holdout_ids = set(int(v) for v in holdout_docs["doc_id"].to_list())
    train_ids = set(int(v) for v in train_docs["doc_id"].to_list())
    overlap = holdout_ids & train_ids
    assert not overlap, f"BUG: {len(overlap)} holdout ids overlap training: {sorted(list(overlap))[:5]}"
    logger.info("Holdout/train overlap check: ∅ (assertion passed)")

    # Transductive context = the union of all known documents at inference time.
    # Holdout rows alone would yield empty neighbour caches because the training
    # graph holds most of the citation density.
    combined_docs = pl.concat([train_docs, holdout_docs], how="vertical_relaxed")

    logger.info("Loading encoder cache from {}", encoder_path)
    encoders, encoder_meta = load_encoder_cache(encoder_path)
    logger.info("Encoder sizes: {}", {k: len(v) for k, v in encoders.items()})

    tokenizer = create_tokenizer(model_cfg["tokenizer_name"])

    # Build a combined citations parquet under the holdout cache root so the
    # graph loader sees both training edges (for k-hop context) and holdout
    # edges. The file is deterministic across runs; rebuild every call to stay
    # in sync with the holdout directory.
    holdout_cache_root = REPO_ROOT / "cache" / f"{benchmark}_holdout_{args.out_name}"
    holdout_cache_root.mkdir(parents=True, exist_ok=True)
    combined_citations_path = holdout_cache_root / "combined_citations.parquet"
    combined_edges = pl.concat([
        pl.read_parquet(train_citations_path),
        pl.read_parquet(holdout_citations_path)], how="vertical_relaxed").unique()
    combined_edges.write_parquet(combined_citations_path)
    logger.info("Wrote combined citation edges ({} rows) to {}",
                combined_edges.height, combined_citations_path)

    combined_graph = load_citation_graph(
        combined_citations_path, source_col=data_cfg["source_col"],
        target_col=data_cfg["target_col"], node_ids=combined_docs["doc_id"].to_list())
    logger.info("Combined graph: {} nodes, {} edges",
                combined_graph.num_nodes, int(combined_graph.edge_index.shape[1]))

    holdout_node_ids = [int(v) for v in holdout_docs["doc_id"].to_list()]
    holdout_neighbor_cache = build_neighbor_cache(
        combined_graph, holdout_node_ids, combined_docs,
        max_context_size=context_budget(data_cfg),
        valid_node_ids=[int(v) for v in combined_docs["doc_id"].to_list()],
        sampling_strategy=data_cfg["sampling_strategy"],
        connectivity_weight=data_cfg["connectivity_weight"],
        temporal_weight=data_cfg["temporal_weight"],
        reciprocity_weight=data_cfg["reciprocity_weight"],
        overlap_weight=data_cfg["overlap_weight"],
        k_hops=int(data_cfg["k_hops"]),
        spectral_dim=int(data_cfg["spectral_dim"]),
        enable_spectral=bool(data_cfg["enable_spectral"]),
        hub_degree_threshold=int(data_cfg.get("hub_degree_threshold", 0)),
        max_graph_nodes_for_hops=int(data_cfg.get("max_graph_nodes_for_hops", 20_000)),
        n_jobs=int(data_cfg.get("cache_n_jobs", -1)))
    logger.info("Built holdout neighbor cache: {} centers", len(holdout_neighbor_cache))

    # Tokenization: warm-start from the training-time cache when present and
    # tokenize only the holdout-specific rows. The 30k center+context docs from
    # training stay reused as-is.
    train_tokenization_path = REPO_ROOT / project_cfg["cache_dir"] / "tokenized_docs.pt"
    pretokenized: dict[int, dict[str, Any]] = {}
    if train_tokenization_path.exists():
        try:
            pretokenized, _ = load_tokenization_cache(train_tokenization_path)
            logger.info("Reused {} tokenised training docs from {}",
                        len(pretokenized), train_tokenization_path)
        except Exception as exc:
            logger.warning("Could not reuse training tokenization cache ({}); will tokenize on-the-fly.", exc)
            pretokenized = {}
    holdout_tokens = build_tokenization_cache(
        holdout_docs, tokenizer, max_seq_length=int(data_cfg["max_seq_length"]))
    pretokenized = {**pretokenized, **holdout_tokens}
    logger.info("Pretokenised lookup size after merge: {}", len(pretokenized))

    # max_authors mirrors the per-split logic in build_run_bundle: derive from
    # the dataset's own document slice unless the config pins a fixed value.
    # NB: build_dataset's type hint says pd.DataFrame but the dataset internally
    # calls .clone() which only exists on polars frames — match production usage
    # in build_run_bundle (which always passes polars) and pass polars here too.
    data_cfg_local = dict(data_cfg)
    if data_cfg_local.get("max_authors") is None:
        data_cfg_local["max_authors"] = infer_max_authors(holdout_docs.to_pandas())

    dataset = build_dataset(
        docs=holdout_docs,
        context_docs=combined_docs,
        tokenizer=tokenizer, encoders=encoders,
        context_cache=holdout_neighbor_cache, data_cfg=data_cfg_local,
        pretokenized=pretokenized)

    holdout_loader = build_loader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=int(cfg.get("train", {}).get("num_workers", 0)))

    bundle: dict[str, Any] = {
        "num_classes": num_classes, "encoders": encoders,
        "label_names": label_names, "graph": combined_graph,
        "labeled_prior": labeled_prior(train_docs, num_classes)}
    model = build_model(bundle, {**cfg, "data": data_cfg_local})

    trainer_cfg = {
        **trainer_cfg_in,
        "output_dir": str(run_dir),  # so load_checkpoint finds checkpoints/best_model.pt
        "ablation_mode": args.ablation,
        "run_name": f"{benchmark}_{args.ablation}_seed_{args.seed}_{args.out_name}",
        "use_mlflow": False, "use_wandb": False,
        # Force fp32 on CPU; bf16/fp16 accelerator paths require CUDA.
        "mixed_precision": "no" if device == "cpu" else trainer_cfg_in.get("mixed_precision", "no")}

    trainer = MetaGraphSciTrainerEval(
        model=model, citation_graph=combined_graph,
        neighbor_cache={k: {int(e["doc_id"]) for e in v} for k, v in holdout_neighbor_cache.items()},
        config=trainer_cfg, label_names=label_names,
        labeled_class_prior=bundle["labeled_prior"])

    # Move the model to the right device BEFORE loading the checkpoint so the
    # state-dict tensors land on the matching device with no extra copy.
    trainer.model = trainer.accelerator.prepare(trainer.model)
    holdout_loader = trainer.accelerator.prepare(holdout_loader)

    logger.info("Loading checkpoint from {}", trainer.best_checkpoint)
    trainer.load_checkpoint()
    logger.info("Checkpoint loaded; running inference on {} docs (device={})",
                holdout_docs.height, trainer.device)

    artifact_dir = run_dir / "artifacts" / args.out_name
    result = trainer.evaluate(
        holdout_loader, split=args.out_name,
        return_embeddings=True, output_dir=artifact_dir)

    metrics = result["metrics"]
    metrics_path = run_dir / f"{args.out_name}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote metrics -> {}", metrics_path)
    logger.info("Metrics summary: {}", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))})

    # Per-paper CSV with title + top-3 predicted labels. The bundle's
    # predictions table already has doc_id/label/prediction/confidence/correct,
    # but it lacks titles and top-k probabilities, which are the bits humans
    # actually want to skim during OOD inspection.
    import numpy as np

    y_prob: np.ndarray = result["y_prob"]
    doc_ids = result["doc_ids"].tolist()
    y_true = result["y_true"].tolist()
    y_pred = result["y_pred"].tolist()

    title_lookup = {int(row["doc_id"]): str(row["title"])
                    for row in holdout_docs.iter_rows(named=True)}

    top_k = min(3, y_prob.shape[1])
    top_idx = np.argsort(-y_prob, axis=1)[:, :top_k]
    rows: list[dict[str, Any]] = []
    for row_idx, doc_id in enumerate(doc_ids):
        top_indices = top_idx[row_idx].tolist()
        rows.append({
            "doc_id": int(doc_id),
            "title": title_lookup.get(int(doc_id), ""),
            "y_true": int(y_true[row_idx]),
            "y_pred": int(y_pred[row_idx]),
            "y_true_name": label_names[int(y_true[row_idx])],
            "y_pred_name": label_names[int(y_pred[row_idx])],
            "prob_top1": float(y_prob[row_idx, top_indices[0]]),
            "prob_top2": float(y_prob[row_idx, top_indices[1]]) if top_k > 1 else float("nan"),
            "prob_top3": float(y_prob[row_idx, top_indices[2]]) if top_k > 2 else float("nan"),
            "top3_labels": "|".join(label_names[i] for i in top_indices)})

    predictions_path = run_dir / f"{args.out_name}_predictions.csv"
    pl.DataFrame(rows).write_csv(predictions_path)
    logger.info("Wrote per-paper predictions -> {}", predictions_path)
    logger.info("Done. Inspect {} and {} for the headline results.", metrics_path, artifact_dir)


if __name__ == "__main__":
    main()
