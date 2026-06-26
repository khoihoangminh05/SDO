#!/usr/bin/env python3
"""Aggregate ablation metrics into a comparison table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/ablation")
    parser.add_argument("--output", default="results/ablation_comparison_table.md")
    args = parser.parse_args()

    results_dir = ROOT / args.results_dir
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*_metrics.json")):
        with path.open(encoding="utf-8") as handle:
            rows.append({"variant": path.stem, **json.load(handle)})

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| Variant | AP50 | APsmall | Recall | Precision | FPR |", "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row.get('variant', '?')} | "
            f"{row.get('ap50', '—')} | {row.get('apsmall', '—')} | "
            f"{row.get('recall', '—')} | {row.get('precision', '—')} | {row.get('fpr', '—')} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
