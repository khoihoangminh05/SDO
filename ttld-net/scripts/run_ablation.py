#!/usr/bin/env python3
"""Run M0–M4 ablation experiments sequentially (Phase 7)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = [
    "m0_baseline",
    "m1_shallow",
    "m2_focal",
    "m3_topology",
    "m4_full_ttld",
]


def _run(cmd: list[str], dry_run: bool) -> None:
    line = " ".join(cmd)
    print(line, flush=True)
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 ablation runner (M0–M4)")
    parser.add_argument("--configs", nargs="*", default=CONFIGS, help="Subset of config stems")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Enable AMP (default: off)")
    parser.add_argument("--max-steps", type=int, default=None, help="Smoke test: stop after N steps")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override epochs in config")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Fast ablation profile (~1h/model on 4090)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    use_no_amp = args.amp and not args.fast

    for name in args.configs:
        config = ROOT / "configs" / f"{name}.yaml"
        if not config.is_file():
            print(f"SKIP missing config: {config}")
            continue

        print(f"\n{'=' * 60}\nAblation: {name}\n{'=' * 60}", flush=True)

        if not args.skip_train:
            train_cmd = [
                sys.executable,
                "train.py",
                "--config",
                str(config),
                "--output",
                f"logs/ablation/{name}",
                "--device",
                str(args.device),
                "--batch-size",
                str(args.batch_size),
            ]
            if args.fast:
                train_cmd.append("--fast")
            elif use_no_amp:
                train_cmd.append("--no-amp")
            if args.max_steps is not None:
                train_cmd.extend(["--max-steps", str(args.max_steps)])
            if args.max_epochs is not None:
                train_cmd.extend(["--max-epochs", str(args.max_epochs)])
            _run(train_cmd, args.dry_run)

        if not args.skip_test:
            weights = ROOT / "checkpoints" / f"{name}_best.pth"
            if name == "m0_baseline":
                weights = ROOT / "checkpoints" / "m0_baseline_best.pt"
            test_cmd = [
                sys.executable,
                "test.py",
                "--config",
                str(config),
                "--weights",
                str(weights),
                "--output",
                f"results/ablation/{name}_metrics.json",
                "--device",
                str(args.device),
            ]
            _run(test_cmd, args.dry_run)

    if not args.skip_test and not args.dry_run:
        compare_cmd = [
            sys.executable,
            "scripts/compare_ablation.py",
            "--results-dir",
            "results/ablation",
        ]
        _run(compare_cmd, dry_run=False)

    print("\nPhase 7 ablation run finished.", flush=True)


if __name__ == "__main__":
    main()
