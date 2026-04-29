# Train MetaGraphSci on data\openalex_ai using an Intel GPU (XPU) on Windows.
# Prereq: scripts\setup_intel_windows.ps1 already ran successfully.

$ErrorActionPreference = "Stop"

if (Test-Path .\.venv\Scripts\Activate.ps1) {
    & .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "No .venv found. Run scripts\setup_intel_windows.ps1 first." -ForegroundColor Red
    exit 1
}

$env:CONFIG = if ($env:CONFIG) { $env:CONFIG } else { "configs/openalex_ai_intel.yaml" }

# Force the trainer onto the Intel XPU. Accelerate >=0.30 reads this var
# and bypasses CUDA detection.
$env:ACCELERATE_USE_XPU         = "true"
$env:ACCELERATE_MIXED_PRECISION = "bf16"

# DataLoader on Windows: keep workers at 0 (already in YAML) to avoid
# multiprocessing fork issues. Disable HF tokenizer parallelism warnings.
$env:TOKENIZERS_PARALLELISM = "false"
# Some transformer ops are CPU-only; let them fall back gracefully.
$env:PYTORCH_ENABLE_XPU_FALLBACK = "1"

python -c "import torch; assert torch.xpu.is_available(), 'Intel XPU not detected. Reinstall drivers + run setup script.'; print('[xpu]', torch.xpu.get_device_name(0))"

python -m src.pipeline --config $env:CONFIG
exit $LASTEXITCODE
