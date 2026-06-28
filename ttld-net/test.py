#!/usr/bin/env python3
"""TTLD-Net evaluation entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config, resolve_path
from utils.metrics import save_metrics
from utils.yolo_baseline import evaluate_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TTLD-Net")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--output", required=True, help="Metrics JSON output path")
    parser.add_argument("--conf", type=float, default=None, help="Override eval confidence threshold")
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    weights = resolve_path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    device: str | int = int(args.device) if args.device.isdigit() else args.device

    if cfg.model.mode == "baseline":
        metrics = evaluate_baseline(weights, cfg, conf=args.conf, device=device)
        save_metrics(metrics, args.output)
        print("Evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print(f"\nSaved: {args.output}")
        return

    raise NotImplementedError(f"Evaluation mode '{cfg.model.mode}' not implemented yet.")


if __name__ == "__main__":
    main()
