#!/usr/bin/env bash
# Phase 0 minimal setup when disk space is limited (~8 GB free needed)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Disk check ==="
df -h ~ /tmp | head -5
FREE_GB=$(df -BG ~ | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
if [ "${FREE_GB:-0}" -lt 10 ]; then
  echo "WARN: Less than 10 GB free. Run cleanup first:"
  echo "  conda clean -a -y"
  echo "  pip cache purge"
  echo "  rm -rf ~/.cache/pip"
fi

echo "=== Cleanup caches ==="
conda clean -a -y 2>/dev/null || true
pip cache purge 2>/dev/null || true

echo "=== Create env (conda only, no pip torch duplicate) ==="
if conda env list | grep -q "^ttld-net "; then
  echo "Env ttld-net exists — activate and skip create, or: conda env remove -n ttld-net"
else
  conda env create -f environment.yml
fi

echo "=== Activate ==="
echo "Run: conda activate ttld-net"

echo "=== Install MMCV ==="
echo "Run: python scripts/install_mmcv.py --run"

echo "=== Phase 0 gate ==="
echo "Run: python test_env.py --strict-gpu"
