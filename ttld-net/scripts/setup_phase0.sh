#!/usr/bin/env bash
# Phase 0 setup for Linux GPU server
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== TTLD-Net Phase 0 Setup ==="

# T0.1 — CUDA check
echo "--- T0.1 CUDA Environment ---"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "WARN: nvidia-smi not found"
fi

if command -v nvcc &>/dev/null; then
  nvcc --version
fi

# T0.2 — Core deps
echo "--- T0.2 Installing core dependencies ---"
pip install -r requirements.txt
pip install pytest

# Detect torch CUDA version for MMCV wheel
TORCH_VER=$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")
CUDA_VER=$(python -c "import torch; v=torch.version.cuda; print('cu'+v.replace('.','')[:3] if v else 'cpu')")
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Detected: torch=${TORCH_VER}, cuda_tag=${CUDA_VER}"

# T0.3 — MMCV with CUDA
echo "--- T0.3 Installing MMCV (CUDA kernel) ---"
MMCV_URL="https://download.openmmlab.com/mmcv/dist/${CUDA_VER}/torch${TORCH_VER}/index.html"
echo "MMCV wheel index: ${MMCV_URL}"
pip install "mmcv==2.1.0" -f "${MMCV_URL}" || {
  echo "WARN: mmcv install failed — try manual URL from SPEC.md §11"
}

# T0.5 — Dataset verify
echo "--- T0.5 Dataset verification ---"
python scripts/verify_dataset.py

# Gate test
echo "--- Phase 0 gate ---"
python test_env.py --strict-gpu | tee logs/env_check.txt
echo "Log saved to logs/env_check.txt"
