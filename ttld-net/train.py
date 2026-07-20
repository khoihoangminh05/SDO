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
from utils.trainer import train_ttld
from utils.yolo_baseline import train_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TTLD-Net")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", default="logs/default", help="Log and checkpoint directory")
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps (smoke test)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)

    device: str | int = int(args.device) if args.device.isdigit() else args.device

    if cfg.model.mode == "baseline":
        best = train_baseline(cfg, output_dir, device=device)
        print("\nM0 baseline training complete.")
        print(f"Best weights: {best}")
        print(
            f"\nNext: python test.py --config {args.config} --weights {best} "
            f"--output results/ablation/m0_baseline_metrics.json"
        )
        return

    best = train_ttld(
        cfg,
        output_dir,
        device=device,
        config_path=config_path,
        resume=args.resume,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
    )
    metrics_out = ROOT / "results" / "ablation" / f"{config_path.stem}_metrics.json"
    print("\nTTLD-Net training complete.")
    print(f"Best checkpoint: {best}")
    print(
        f"\nNext: python test.py --config {args.config} --weights {best} "
        f"--output {metrics_out}"
    )


if __name__ == "__main__":
    main()
