#!/usr/bin/env bash
# Continue Phase 7 ablation from M2 (skip M0 + M1).
# Usage on Colab:
#   cd /content/SDO/ttld-net && bash scripts/continue_m2.sh
# Or with Pro+ profile (default):
#   bash scripts/continue_m2.sh --proplus
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROFILE="--proplus"
DEVICE="0"
CONFIGS=("m2_focal")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) PROFILE="--fast"; shift ;;
    --proplus) PROFILE="--proplus"; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --through-m4) CONFIGS=("m2_focal" "m3_topology" "m4_full_ttld"); shift ;;
    -h|--help)
      echo "Usage: $0 [--proplus|--fast] [--device ID] [--through-m4]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "Continue ablation: ${CONFIGS[*]}  profile=${PROFILE}  device=${DEVICE}"
echo "Keeping M0/M1 checkpoints untouched."
echo "============================================================"

# Do not delete m1_shallow_best.pth — already validated.
python scripts/run_ablation.py \
  --configs "${CONFIGS[@]}" \
  --device "${DEVICE}" \
  ${PROFILE}

echo "Done. Metrics under results/ablation/"
ls -la results/ablation/m2_focal_metrics.json 2>/dev/null || true
