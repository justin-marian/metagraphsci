"""Fast inference-only eval of a trained MetaGraphSci checkpoint on a holdout slice.

Stripped-down sibling of ``eval_holdout.py`` for situations where the full
preprocessing pipeline is too slow (the upstream ``local_relevance_func`` is
the dominant cost — ~9 min for 100 centers on a 30k-node graph).

Differences vs ``eval_holdout.py``:
- No combined graph build, no neighbor-cache build. Every holdout paper sees
  an empty citation context (all-padding context tensors). For holdouts that
  already had 0 internal citation edges this changes nothing; otherwise it
  measures the model's behaviour with the citation pathway ablated to padding.
- No tSNE/UMAP/PCA embedding plots and no per-class diagnostic plots beyond
  what ``trainer.evaluate(output_dir=None)`` produces by default (which is none).
- Still loads the same checkpoint, the same encoders, applies the same
  training label-id mapping, and uses the same ``trainer.evaluate()`` codepath
  so the headline metrics are directly comparable to the slow eval's numbers
  (modulo the ablated citation context).

Usage (run from repo root):
    python scripts/eval_holdout_fast.py \\
        --config configs/openalex_ai_rtx5090_mid_1seed.yaml \\
        --holdout-dir data/openalex_ai_holdout100 \\
        --ablation full --seed 42 --device cuda

Expected wall-clock: 1.5-3 min on GPU (vs 11-13 min for the full eval).
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
    parser = argparse.ArgumentParser(description="Fast inference-only eval (no graph context).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, default=Path("data/openalex_ai"))
    parser.add_argument("--ablation", type=str, default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-name", type=str, default="holdout100_eval_fast")
    return parser.parse_args()


def force_device_env(device: str) -> str:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    import torch
    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    return "cuda"


def normalise_holdout(raw_holdout, label_names: list[str], label_column: str = "label"):
    """Apply training label-id mapping while reusing the canonical normalisation."""
    import polars as pl
    from src.data.tabular_utils import prepare_documents

    if label_column not in raw_holdout.columns:
        raise ValueError(f"Holdout missing required '{label_column}' column.")
    raw_strings = [str(v) if v is not None else None
                   for v in raw_holdout[label_column].cast(pl.String).to_list()]
    mapping = {name: idx for idx, name in enumerate(label_names)}
    unknown = sorted(set(s for s in raw_strings if s is not None) - set(label_names))
    if unknown:
        raise ValueError(f"Holdout contains {len(unknown)} unknown labels: {unknown[:5]}")

    stripped = raw_holdout.drop(label_column)
    normalised, _ = prepare_documents(stripped, label_column=label_column)
    ids = [mapping[s] if s is not None else None for s in raw_strings]
    return normalised.with_columns(pl.Series(name="label", values=ids, dtype=pl.Int64))


def main() -> None:
    args = parse_args()
    print(f"[eval_holdout_fast] starting; config={args.config} device={args.device}", flush=True)
    device = force_device_env(args.device)
    print(f"[eval_holdout_fast] device={device}; importing torch ...", flush=True)

    from loguru import logger
    import sys as _sys
    logger.remove()
    logger.add(_sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    import polars as pl
    import torch
    import numpy as np

    from src.data import build_loader, create_tokenizer, load_documents, load_encoder_cache
    from src.pipeline import build_dataset, build_model, infer_max_authors, labeled_prior, read_config
    from src.train_eval import MetaGraphSciTrainerEval

    print(f"[eval_holdout_fast] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

    holdout_docs_path = args.holdout_dir / "documents.parquet"
    train_docs_path = args.train_dir / "documents.parquet"
    for path in (args.config, holdout_docs_path, train_docs_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    cfg = read_config(args.config)
    project_cfg, data_cfg, model_cfg = cfg["project"], cfg["data"], cfg["model"]
    trainer_cfg_in = dict(cfg.get("trainer", {}))

    benchmark = project_cfg["benchmark"]
    run_dir = REPO_ROOT / project_cfg["output_dir"] / benchmark / args.ablation / f"seed_{args.seed}"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint missing at {checkpoint_path}")

    encoder_path = REPO_ROOT / project_cfg["cache_dir"] / f"seed_{args.seed}" / "encoders.json"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder cache missing at {encoder_path}")

    logger.info("Loading training docs for label mapping")
    train_docs, label_names = load_documents(str(train_docs_path), label_column=data_cfg["label_column"])
    num_classes = len(label_names)
    logger.info("Loaded {} training docs, {} classes", train_docs.height, num_classes)

    logger.info("Loading + normalising holdout docs")
    raw_holdout = pl.read_parquet(holdout_docs_path)
    holdout_docs = normalise_holdout(raw_holdout, label_names, data_cfg["label_column"])
    overlap = set(holdout_docs["doc_id"].to_list()) & set(train_docs["doc_id"].to_list())
    assert not overlap, f"BUG: {len(overlap)} overlapping ids"
    logger.info("Holdout: {} docs (overlap with train: ∅)", holdout_docs.height)

    logger.info("Loading encoder cache")
    encoders, _ = load_encoder_cache(encoder_path)
    logger.info("Encoders: {}", {k: len(v) for k, v in encoders.items()})

    logger.info("Building tokenizer")
    tokenizer = create_tokenizer(model_cfg["tokenizer_name"])

    # FAST PATH: skip the heavy graph build + neighbor cache scoring entirely.
    # Empty neighbour cache means each sample's context tensor is all-padding,
    # which is what the model receives anyway when a paper has no graph
    # neighbours (the holdout slice already had 0 internal citation edges).
    empty_neighbor_cache: dict[int, list[dict[str, Any]]] = {}

    data_cfg_local = dict(data_cfg)
    if data_cfg_local.get("max_authors") is None:
        data_cfg_local["max_authors"] = infer_max_authors(holdout_docs.to_pandas())
    # The dataset still respects pretokenize_context but with no context docs
    # to pretokenize the call is a no-op; keep the setting as-is.

    logger.info("Building dataset (no graph context)")
    dataset = build_dataset(
        docs=holdout_docs,
        context_docs=holdout_docs,  # self as context; ignored anyway with empty cache
        tokenizer=tokenizer, encoders=encoders,
        context_cache=empty_neighbor_cache, data_cfg=data_cfg_local,
        pretokenized=None)
    loader = build_loader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info("Building model + trainer")
    bundle: dict[str, Any] = {
        "num_classes": num_classes, "encoders": encoders,
        "label_names": label_names, "graph": None,
        "labeled_prior": labeled_prior(train_docs, num_classes)}
    model = build_model(bundle, {**cfg, "data": data_cfg_local})

    trainer_cfg = {
        **trainer_cfg_in,
        "output_dir": str(run_dir),
        "ablation_mode": args.ablation,
        "run_name": f"{benchmark}_{args.ablation}_seed_{args.seed}_{args.out_name}",
        "use_mlflow": False, "use_wandb": False,
        "mixed_precision": "no" if device == "cpu" else trainer_cfg_in.get("mixed_precision", "no")}

    trainer = MetaGraphSciTrainerEval(
        model=model, citation_graph=None, neighbor_cache={},
        config=trainer_cfg, label_names=label_names,
        labeled_class_prior=bundle["labeled_prior"])
    trainer.model = trainer.accelerator.prepare(trainer.model)
    loader = trainer.accelerator.prepare(loader)

    logger.info("Loading checkpoint from {}", trainer.best_checkpoint)
    trainer.load_checkpoint()
    logger.info("Running inference on {} docs (device={})", holdout_docs.height, trainer.device)

    # output_dir=None skips the heavy artifact bundle (no plots, no UMAP). We
    # still get y_true / y_pred / y_prob / doc_ids back for our own CSV dump.
    result = trainer.evaluate(loader, split=args.out_name, return_embeddings=False, output_dir=None)

    metrics_path = run_dir / f"{args.out_name}.json"
    metrics_path.write_text(json.dumps(result["metrics"], indent=2))
    logger.info("Wrote metrics -> {}", metrics_path)
    logger.info("Metrics: {}", {k: round(v, 4) for k, v in result["metrics"].items()
                                if isinstance(v, (int, float))})

    title_lookup = {int(row["doc_id"]): str(row["title"])
                    for row in holdout_docs.iter_rows(named=True)}
    y_prob = np.asarray(result["y_prob"])
    top_k = min(3, y_prob.shape[1])
    top_idx = np.argsort(-y_prob, axis=1)[:, :top_k]
    rows = []
    for row_idx, doc_id in enumerate(result["doc_ids"].tolist()):
        ti = top_idx[row_idx].tolist()
        rows.append({
            "doc_id": int(doc_id),
            "title": title_lookup.get(int(doc_id), ""),
            "y_true": int(result["y_true"][row_idx]),
            "y_pred": int(result["y_pred"][row_idx]),
            "y_true_name": label_names[int(result["y_true"][row_idx])],
            "y_pred_name": label_names[int(result["y_pred"][row_idx])],
            "prob_top1": float(y_prob[row_idx, ti[0]]),
            "prob_top2": float(y_prob[row_idx, ti[1]]) if top_k > 1 else float("nan"),
            "prob_top3": float(y_prob[row_idx, ti[2]]) if top_k > 2 else float("nan"),
            "top3_labels": "|".join(label_names[i] for i in ti)})
    predictions_path = run_dir / f"{args.out_name}_predictions.csv"
    pl.DataFrame(rows).write_csv(predictions_path)
    logger.info("Wrote predictions -> {}", predictions_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
