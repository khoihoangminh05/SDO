#!/usr/bin/env python3
"""Run M0–M4 ablation experiments sequentially."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIGS = [
    "m0_baseline",
    "m1_shallow",
    "m2_focal",
    "m3_topology",
    "m4_full_ttld",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=CONFIGS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for name in args.configs:
        config = ROOT / "configs" / f"{name}.yaml"
        print(f"\n{'=' * 50}\nAblation: {name}\n{'=' * 50}")
        train_cmd = (
            f'python train.py --config "{config}" --output logs/ablation/{name}'
        )
        test_cmd = (
            f'python test.py --config "{config}" '
            f'--weights checkpoints/{name}_best.pth '
            f'--output results/ablation/{name}_metrics.json'
        )
        print(train_cmd)
        print(test_cmd)
        if not args.dry_run:
            os.system(train_cmd)
            os.system(test_cmd)


if __name__ == "__main__":
    main()
