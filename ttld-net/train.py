#!/usr/bin/env python3
"""TTLD-Net training entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config
from utils.yolo_baseline import train_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TTLD-Net")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", default="logs/default", help="Log and checkpoint directory")
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.model.mode == "baseline":
        device: str | int = int(args.device) if args.device.isdigit() else args.device
        best = train_baseline(cfg, output_dir, device=device)
        print(f"\nM0 baseline training complete.")
        print(f"Best weights: {best}")
        print(f"\nNext: python test.py --config {args.config} --weights {best} "
              f"--output results/ablation/m0_baseline_metrics.json")
        return

    raise NotImplementedError(
        f"Training mode '{cfg.model.mode}' not implemented yet. "
        "Phase 1 supports mode=baseline only."
    )


if __name__ == "__main__":
    main()
