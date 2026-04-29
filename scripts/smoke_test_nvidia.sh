#!/usr/bin/env bash
# End-to-end smoke test for MetaGraphSci on a single NVIDIA GPU.
# Runs 1 seed x 1 ablation x 1+1 epochs to validate the full pipeline before
# committing GPU hours to the full experiment matrix.
#
# Usage on RunPod (or any CUDA box):
#   bash scripts/smoke_test_nvidia.sh
#
# Expected wall time: ~15-30 min on H100/H200, ~30-60 min on A100/4090.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/openalex_ai_smoke.yaml}"

echo "=== MetaGraphSci :: Smoke test ==="
echo "Config: $CONFIG"

# Fail fast if the GPU is not visible.
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA GPU not visible to PyTorch. On RunPod pick a GPU template."
name = torch.cuda.get_device_name(0)
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"[gpu] {name}  CUDA={torch.version.cuda}  bf16={torch.cuda.is_bf16_supported()}  mem={mem_gb:.1f} GB")
PY

# Fail fast if the dataset is missing.
python - <<'PY'
from pathlib import Path
required = [
    "data/openalex_ai/documents.parquet",
    "data/openalex_ai/citations.parquet",
]
missing = [p for p in required if not Path(p).is_file()]
assert not missing, f"Missing dataset files: {missing}. Place data/openalex_ai/ on the pod first."
PY

# Use bf16 automatically on Ampere+ (A100/H100/H200/RTX 30/40), fp16 otherwise.
MIXED_PRECISION="$(python -c 'import torch; print("bf16" if torch.cuda.is_bf16_supported() else "fp16")')"
export ACCELERATE_MIXED_PRECISION="$MIXED_PRECISION"
echo "[precision] $MIXED_PRECISION"

START_TS=$(date +%s)
echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ)"

python -m src.pipeline --config "$CONFIG"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
printf '[done] %s  elapsed=%dm%02ds\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" $((ELAPSED / 60)) $((ELAPSED % 60))

# Surface the run summary so the user can eyeball metrics without digging.
SUMMARY="runs/openalex_ai_smoke/openalex_ai/full/seed_42/artifacts/run_summary.json"
if [ -f "$SUMMARY" ]; then
    echo "=== Smoke test summary ==="
    cat "$SUMMARY"
else
    echo "WARNING: expected summary not found at $SUMMARY"
    exit 1
fi

echo
echo "Smoke test passed. To launch the full matrix:"
echo "    bash scripts/train_nvidia.sh"
