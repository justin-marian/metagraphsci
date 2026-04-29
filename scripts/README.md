# Training MetaGraphSci on `data/openalex_ai`

Two parallel setups are provided. They share the same code path
(`src/pipeline.py`) and differ only in config + launch wrapper.

| Variant | Hardware | Config | Launcher |
|---|---|---|---|
| NVIDIA / RunPod (Linux) | CUDA GPU (RTX 30/40, A100, H100) | [`configs/openalex_ai_nvidia.yaml`](../configs/openalex_ai_nvidia.yaml) | [`scripts/train_nvidia.sh`](train_nvidia.sh) |
| Intel GPU (Windows) | Intel Arc / Battlemage / Iris Xe | [`configs/openalex_ai_intel.yaml`](../configs/openalex_ai_intel.yaml) | [`scripts/train_intel.ps1`](train_intel.ps1) |

Outputs: `runs/openalex_ai_<variant>/` ; cache: `cache/openalex_ai/`.

---

## NVIDIA — local Linux box or RunPod

Local:
```bash
bash scripts/train_nvidia.sh
```

RunPod: see [`docker/runpod_README.md`](../docker/runpod_README.md).
The image at [`docker/Dockerfile.runpod`](../docker/Dockerfile.runpod) bakes the
project + deps onto `pytorch/pytorch:2.4.1-cuda12.4-cudnn9`.

The launcher auto-picks `bf16` on Ampere+ GPUs and falls back to `fp16` on Turing
or older. Tune `batch_size` in the YAML to match VRAM (comments inline).

## Intel — Windows

Driver + runtime prerequisite: install the latest **Intel Arc Graphics driver**
and **Intel oneAPI runtime** (Arc cards ship with this; Iris Xe needs the latest
GPU driver from Intel).

First-time setup:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_intel_windows.ps1
```
The setup script:
1. Creates `.venv` (Python 3.12).
2. Installs PyTorch with native XPU support (`--index-url …/whl/xpu`).
3. Installs the rest of `requirements.txt`, **skipping** CUDA-only packages
   (`bitsandbytes`, `deepspeed`).
4. Verifies `torch.xpu.is_available()`.

Train:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_intel.ps1
```

The Intel config trades throughput for memory:
- `max_seq_length: 256` (vs 512), `max_context_size: 12` (vs 18)
- `batch_size: 4` + `gradient_accumulation_steps: 4` → effective batch 16
- `bf16` mixed precision (Arc/Battlemage have native bf16)
- `num_workers: 0` (Windows DataLoader)
- A single seed + the `full` ablation (avoid 12-run sweep on consumer hardware)

If you have an Arc A770 16GB, raise `batch_size` to 8 and drop `grad_accum` to 2.
