#!/usr/bin/env python3
"""TTLD-Net training entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure ttld-net/ is on sys.path when run from repo root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TTLD-Net")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", default="logs/default", help="Log and checkpoint directory")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "Phase 6 (T6.1): implement training loop. "
        f"Config={args.config}, output={output_dir}, mode={cfg.model.mode}"
    )


if __name__ == "__main__":
    main()
