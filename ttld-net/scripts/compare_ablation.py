#!/usr/bin/env python3
"""Aggregate ablation metrics into a Markdown comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COLUMNS = [
    ("variant", "Variant"),
    ("ap50", "AP50"),
    ("map50_95", "mAP50-95"),
    ("apsmall", "APsmall"),
    ("ap_tiny", "APtiny"),
    ("recall", "Recall"),
    ("precision", "Precision"),
    ("fpr", "FPR"),
    ("tp", "TP"),
    ("fp", "FP"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ablation comparison table")
    parser.add_argument("--results-dir", default="results/ablation")
    parser.add_argument("--output", default="results/ablation_comparison_table.md")
    args = parser.parse_args()

    results_dir = ROOT / args.results_dir
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*_metrics.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append({"variant": path.stem.replace("_metrics", ""), **payload})

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [
        "# Ablation comparison",
        "",
        "_AP50 / mAP50-95 / size-bin AP are VOC-style averages. "
        "Precision/Recall are at the configured operating confidence._",
        "",
        header,
        sep,
    ]
    for row in rows:
        cells = []
        for key, _ in COLUMNS:
            val = row.get(key, "—")
            if isinstance(val, float):
                cells.append(f"{val:.4f}" if abs(val) < 10 else f"{val:.1f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    if not rows:
        lines.append("| _(no metrics JSON found)_ | " + " | ".join("—" for _ in COLUMNS[1:]) + " |")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} variants)")


if __name__ == "__main__":
    main()
