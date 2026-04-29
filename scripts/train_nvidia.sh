#!/usr/bin/env bash
# Train MetaGraphSci on data/openalex_ai using a single NVIDIA GPU.
# Works on local CUDA boxes and RunPod pods (PyTorch templates).
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/openalex_ai_nvidia.yaml}"

# Fail fast if the GPU is not visible.
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA GPU not visible to PyTorch. On RunPod pick a GPU template."
print(f"[gpu] {torch.cuda.get_device_name(0)}  CUDA={torch.version.cuda}  bf16={torch.cuda.is_bf16_supported()}")
PY

# Use bf16 automatically on Ampere+ (A100/H100/RTX 30/40), fp16 otherwise.
MIXED_PRECISION="$(python -c 'import torch; print("bf16" if torch.cuda.is_bf16_supported() else "fp16")')"
export ACCELERATE_MIXED_PRECISION="$MIXED_PRECISION"

# Multi-GPU: launch with `accelerate launch --multi_gpu` instead.
exec python -m src.pipeline --config "$CONFIG"
