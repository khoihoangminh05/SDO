#!/usr/bin/env python3
"""TTLD-Net evaluation entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config
from utils.metrics import compute_metrics, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TTLD-Net")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--output", required=True, help="Metrics JSON output path")
    parser.add_argument("--conf", type=float, default=None, help="Override eval confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    conf = args.conf if args.conf is not None else cfg.data.eval_conf_threshold

    raise NotImplementedError(
        "Phase 1 (T1.6) / Phase 6: implement evaluation. "
        f"weights={args.weights}, conf={conf}, output={args.output}"
    )


if __name__ == "__main__":
    main()
