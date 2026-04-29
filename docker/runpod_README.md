# Running on RunPod

## Option A — Build the image and push to a registry
```bash
docker build -f docker/Dockerfile.runpod -t <youruser>/metagraphsci:latest .
docker push <youruser>/metagraphsci:latest
```
On RunPod create a Pod (any NVIDIA GPU: RTX 4090 / A100 / H100), set:
- **Container image**: `<youruser>/metagraphsci:latest`
- **Container start command**: `bash scripts/train_nvidia.sh`
- **Volume mount**: `/workspace` (persistent storage for the dataset + outputs)

Upload `data/openalex_ai/*.parquet` to `/workspace/metagraphsci/data/openalex_ai/`
(or bake them into the image — the parquet files are kept in-repo).

## Option B — Use a stock PyTorch template (no Docker build)
1. Pick the official "PyTorch 2.4 / CUDA 12.4" RunPod template.
2. In the pod terminal:
   ```bash
   cd /workspace
   git clone <your-repo-url> metagraphsci && cd metagraphsci
   pip install -r requirements.txt && pip install -e .
   bash scripts/train_nvidia.sh
   ```
3. Outputs land in `runs/openalex_ai_nvidia/` and `cache/openalex_ai/`.

## GPU sizing
The config defaults to `batch_size: 16` (fits a 24GB 4090/3090). For larger pods edit
`configs/openalex_ai_nvidia.yaml`:
- A100 40GB → `batch_size: 32`, `gradient_accumulation_steps: 1`
- A100/H100 80GB → `batch_size: 48`, `gradient_accumulation_steps: 1`
