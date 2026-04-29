# Setup Intel GPU (Arc / Battlemage / Iris Xe) training environment on Windows.
# PowerShell. Run in the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_intel_windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== MetaGraphSci :: Intel GPU (XPU) Windows setup ===" -ForegroundColor Cyan

# 1) Python venv
if (-Not (Test-Path .venv)) {
    Write-Host "[1/4] Creating virtual env (.venv)..." -ForegroundColor Yellow
    py -3.12 -m venv .venv
}
& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip wheel

# 2) PyTorch with native Intel XPU support (PyTorch >= 2.5).
# Reference: https://pytorch.org/get-started/locally/  (select "XPU")
Write-Host "[2/4] Installing PyTorch + XPU runtime..." -ForegroundColor Yellow
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# 3) Project deps (skip torch lines so the XPU build above is preserved,
#    skip bitsandbytes/deepspeed which are CUDA-only).
Write-Host "[3/4] Installing project dependencies..." -ForegroundColor Yellow
$req = Get-Content requirements.txt | Where-Object {
    $_ -notmatch '^(torch|torchvision|torchaudio|bitsandbytes|deepspeed)\s*$'
}
$req | Set-Content requirements.intel.txt
pip install -r requirements.intel.txt
# --no-deps: setup.py's install_requires reads the unfiltered requirements.txt,
# which would re-pull deepspeed / bitsandbytes (CUDA-only, broken on Windows).
# Deps were already installed from requirements.intel.txt above.
pip install -e . --no-deps

# 4) Verify XPU is visible
Write-Host "[4/4] Verifying Intel XPU runtime..." -ForegroundColor Yellow
python -c "import torch; print('torch', torch.__version__); print('xpu available:', torch.xpu.is_available()); print('device:', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'NONE')"

Write-Host ""
Write-Host "Setup complete. Run training with:" -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\train_intel.ps1" -ForegroundColor Green
