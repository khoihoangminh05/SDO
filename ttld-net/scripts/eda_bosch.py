#!/usr/bin/env python3
"""Exploratory data analysis for Bosch BSTLD dataset (Phase 1 T1.1)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.bosch_parser import load_bosch_entries
from data.dataset import CLASS_NAMES


def analyze_bbox_distribution(yaml_path: Path, output_dir: Path) -> dict:
    """
    Analyze bbox size distribution, class balance, and tiny-object ratio.

    Saves bbox_distribution.png, class_distribution.png, eda_summary.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_bosch_entries(yaml_path, images_root=yaml_path.parent)

    widths: list[float] = []
    heights: list[float] = []
    class_counts: Counter[str] = Counter()
    boxes_per_image: list[int] = []
    missing_images = 0

    for entry in entries:
        if not entry["image_path"].is_file():
            missing_images += 1
            continue
        boxes = entry["boxes"]
        boxes_per_image.append(len(boxes))
        for box in boxes:
            widths.append(box.width)
            heights.append(box.height)
            class_counts[box.class_name] += 1

    total_boxes = len(widths)
    tiny_w = sum(1 for w in widths if w < 10)
    tiny_h = sum(1 for h in heights if h < 10)
    tiny_both = sum(1 for w, h in zip(widths, heights) if w < 10 and h < 10)

    avg_boxes = float(np.mean(boxes_per_image)) if boxes_per_image else 0.0
    pct_tiny_w = 100.0 * tiny_w / total_boxes if total_boxes else 0.0
    pct_tiny_h = 100.0 * tiny_h / total_boxes if total_boxes else 0.0

    summary = {
        "yaml": str(yaml_path),
        "entries": len(entries),
        "missing_images": missing_images,
        "total_boxes": total_boxes,
        "avg_boxes_per_image": round(avg_boxes, 3),
        "pct_width_lt_10px": round(pct_tiny_w, 2),
        "pct_height_lt_10px": round(pct_tiny_h, 2),
        "pct_both_lt_10px": round(100.0 * tiny_both / total_boxes, 2) if total_boxes else 0.0,
        "width_min": round(min(widths), 2) if widths else 0.0,
        "width_max": round(max(widths), 2) if widths else 0.0,
        "height_min": round(min(heights), 2) if heights else 0.0,
        "height_max": round(max(heights), 2) if heights else 0.0,
        "class_counts": dict(class_counts),
    }

    # Histogram — bbox sizes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(widths, bins=50, color="#3b82f6", edgecolor="white")
    axes[0].axvline(10, color="red", linestyle="--", label="10px")
    axes[0].set_title("Bounding Box Width (px)")
    axes[0].set_xlabel("width")
    axes[0].legend()

    axes[1].hist(heights, bins=50, color="#22c55e", edgecolor="white")
    axes[1].axvline(10, color="red", linestyle="--", label="10px")
    axes[1].set_title("Bounding Box Height (px)")
    axes[1].set_xlabel("height")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bbox_distribution.png", dpi=120)
    plt.close(fig)

    # Class distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    names = CLASS_NAMES
    counts = [class_counts.get(n, 0) for n in names]
    ax.bar(names, counts, color=["#22c55e", "#fbbf24", "#ef4444", "#64748b"])
    ax.set_title("Class Distribution (4-class TTLD mapping)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=120)
    plt.close(fig)

    with (output_dir / "eda_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("BSTLD EDA Summary")
    print("=" * 40)
    print(f"Entries:              {summary['entries']}")
    print(f"Missing images:       {summary['missing_images']}")
    print(f"Total boxes:          {summary['total_boxes']}")
    print(f"Avg boxes/image:      {summary['avg_boxes_per_image']}")
    print(f"% width  < 10px:      {summary['pct_width_lt_10px']}%")
    print(f"% height < 10px:      {summary['pct_height_lt_10px']}%")
    print(f"Width  min/max:       {summary['width_min']} / {summary['width_max']}")
    print(f"Height min/max:       {summary['height_min']} / {summary['height_max']}")
    print("Class counts:", summary["class_counts"])
    print(f"\nSaved: {output_dir / 'bbox_distribution.png'}")
    print(f"Saved: {output_dir / 'class_distribution.png'}")
    print(f"Saved: {output_dir / 'eda_summary.json'}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="BSTLD EDA (Phase 1)")
    parser.add_argument("--yaml", required=True, help="Path to train.yaml")
    parser.add_argument("--output", default="logs", help="Output directory for plots")
    args = parser.parse_args()
    analyze_bbox_distribution(Path(args.yaml), Path(args.output))


if __name__ == "__main__":
    main()
