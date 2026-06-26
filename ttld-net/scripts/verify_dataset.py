#!/usr/bin/env python3
"""T0.5 — Verify Bosch BSTLD dataset paths and YAML format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

DATASET_ROOT = REPO / "apps" / "worker" / "datasets" / "dataset_train_rgb"
TRAIN_YAML = DATASET_ROOT / "train.yaml"
VAL_ROOT = REPO / "apps" / "worker" / "datasets" / "dataset_val_sample"
VAL_YAML = VAL_ROOT / "val.yaml"
VAL_IMAGES = VAL_ROOT / "rgb" / "val"
BSTLD_YAML = REPO / "apps" / "worker" / "datasets" / "bstld.yaml"


def _parse_bstld_list_yaml(yaml_path: Path) -> list[dict]:
    """Parse BSTLD list-format YAML: [{path, boxes}, ...]."""
    with yaml_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected YAML list in {yaml_path}, got {type(data)}")
    return data


def verify_split(yaml_path: Path, images_root: Path, name: str) -> dict:
    """Verify one split and return summary stats."""
    entries = _parse_bstld_list_yaml(yaml_path)
    missing = 0
    with_boxes = 0
    total_boxes = 0
    labels: dict[str, int] = {}

    for entry in entries:
        rel = entry.get("path", "")
        img_path = (images_root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        if not img_path.is_file():
            missing += 1
            continue
        boxes = entry.get("boxes") or []
        if boxes:
            with_boxes += 1
        total_boxes += len(boxes)
        for box in boxes:
            label = box.get("label", "unknown")
            labels[label] = labels.get(label, 0) + 1

    return {
        "name": name,
        "yaml": str(yaml_path),
        "entries": len(entries),
        "missing_images": missing,
        "images_with_boxes": with_boxes,
        "total_boxes": total_boxes,
        "label_counts": labels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify BSTLD dataset (Phase 0 T0.5)")
    parser.add_argument("--output", default=None, help="Optional JSON/text output path")
    args = parser.parse_args()

    print("BSTLD Dataset Verification (T0.5)\n" + "=" * 40)
    ok = True

    for path, label, required in [
        (TRAIN_YAML, "train.yaml (Bosch format)", True),
        (BSTLD_YAML, "bstld.yaml (Ultralytics)", True),
        (VAL_YAML, "val.yaml (Bosch format)", False),
    ]:
        exists = path.is_file()
        mark = "OK" if exists else ("WARN" if not required else "FAIL")
        print(f"[{mark}] {label}: {path}")
        if required and not exists:
            ok = False

    # Val split may use YOLO format (png+txt) via bstld.yaml instead of val.yaml
    if VAL_IMAGES.is_dir():
        val_pngs = list(VAL_IMAGES.glob("*.png"))
        print(f"[OK] val YOLO images: {len(val_pngs)} png in {VAL_IMAGES}")
    else:
        print(f"[WARN] val image dir missing: {VAL_IMAGES}")

    if not TRAIN_YAML.is_file():
        return 1

    train_stats = verify_split(TRAIN_YAML, DATASET_ROOT, "train")
    print(f"\n[train] entries={train_stats['entries']}, "
          f"with_boxes={train_stats['images_with_boxes']}, "
          f"total_boxes={train_stats['total_boxes']}, "
          f"missing_images={train_stats['missing_images']}")

    if train_stats["missing_images"] > 0:
        print(f"WARN: {train_stats['missing_images']} images referenced in YAML not found on disk")
    if train_stats["entries"] == 0:
        print("FAIL: train.yaml has no entries")
        ok = False

    if VAL_YAML.is_file():
        val_stats = verify_split(VAL_YAML, VAL_ROOT, "val")
        print(f"[val] entries={val_stats['entries']}, missing_images={val_stats['missing_images']}")
    labels = train_stats.get("label_counts", {})
    if labels:
        top = sorted(labels.items(), key=lambda x: -x[1])[:5]
        print("Top labels:", ", ".join(f"{k}={v}" for k, v in top))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{k}: {v}" for k, v in train_stats.items() if k != "label_counts"]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
