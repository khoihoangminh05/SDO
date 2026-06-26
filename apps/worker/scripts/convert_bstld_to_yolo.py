"""
Convert the Bosch Small Traffic Lights Dataset (BSTLD) from its native YAML
annotation format into the YOLO label format expected by Ultralytics.

BSTLD native format (per image):
    - boxes:
      - {label: Red, occluded: false, x_min: .., x_max: .., y_min: .., y_max: ..}
      path: ./rgb/train/<bag>/<frame>.png

YOLO format (one .txt next to each image):
    <class_id> <cx> <cy> <w> <h>   # all normalised to [0, 1]

Class mapping (13 BSTLD labels -> 4 classes), keeping red/yellow/green ids
compatible with the existing custom_dataset, plus a new "off" class:
    red=0, yellow=1, green=2, off=3

Images that have no (valid) boxes get an empty .txt file so YOLO treats them as
negative/background samples.

Usage (run from apps/worker):
    # Scan the whole datasets/ tree so images extracted into any sibling folder
    # are found by basename, then write a label .txt next to each image.
    python scripts/convert_bstld_to_yolo.py \
        --yaml datasets/dataset_train_rgb.zip/train.yaml --images-root datasets
    python scripts/convert_bstld_to_yolo.py \
        --yaml datasets/dataset_test_rgb.zip/test.yaml --images-root datasets
"""
import argparse
import glob
import os
import sys
from collections import Counter

import yaml

# BSTLD images are a fixed resolution.
IMG_W = 1280
IMG_H = 720

CLASS_NAMES = {0: "red", 1: "yellow", 2: "green", 3: "off"}

# Map every BSTLD label (including directional variants) to a YOLO class id.
LABEL_TO_CLASS = {
    "Red": 0,
    "RedLeft": 0,
    "RedRight": 0,
    "RedStraight": 0,
    "RedStraightLeft": 0,
    "RedStraightRight": 0,
    "Yellow": 1,
    "Green": 2,
    "GreenLeft": 2,
    "GreenRight": 2,
    "GreenStraight": 2,
    "GreenStraightLeft": 2,
    "GreenStraightRight": 2,
    "off": 3,
}


def build_image_index(roots: list) -> dict:
    """
    Map every PNG basename found under any of ``roots`` to its absolute path.

    Matching by basename is necessary because the BSTLD annotation paths do not
    reliably resolve relative to the dataset folder (test.yaml uses absolute
    Bosch network paths) and because the image zips are extracted into several
    sibling folders. Each frame number is unique, so basename matching is safe.
    """
    index = {}
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "*.png"), recursive=True):
            index.setdefault(os.path.basename(path), path)
    return index


def convert_one(yaml_path: str, image_roots: list = None) -> dict:
    yaml_path = os.path.abspath(yaml_path)
    root = os.path.dirname(yaml_path)
    # Always search the yaml's own folder, plus any extra roots the user passes.
    roots = [root] + [os.path.abspath(r) for r in (image_roots or [])]
    image_index = build_image_index(roots)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    stats = {
        "images": 0,
        "images_missing": 0,
        "labels_written": 0,
        "boxes_total": 0,
        "boxes_kept": 0,
        "boxes_skipped_unknown": 0,
        "boxes_skipped_invalid": 0,
        "class_counts": Counter(),
        "unknown_labels": Counter(),
    }

    for item in data:
        rel_path = item.get("path")
        if not rel_path:
            continue

        # train.yaml uses relative "./rgb/train/<bag>/<frame>.png" paths;
        # test.yaml uses absolute Bosch network paths. Resolve relatively first,
        # then fall back to matching by basename within the dataset tree.
        stats["images"] += 1
        img_path = os.path.normpath(os.path.join(root, rel_path))
        if not os.path.exists(img_path):
            img_path = image_index.get(os.path.basename(rel_path))
        if not img_path or not os.path.exists(img_path):
            stats["images_missing"] += 1
            continue

        lines = []
        for box in item.get("boxes") or []:
            stats["boxes_total"] += 1
            label = box.get("label")
            cls_id = LABEL_TO_CLASS.get(label)
            if cls_id is None:
                stats["boxes_skipped_unknown"] += 1
                stats["unknown_labels"][label] += 1
                continue

            x_min = float(box["x_min"])
            x_max = float(box["x_max"])
            y_min = float(box["y_min"])
            y_max = float(box["y_max"])

            # Normalise ordering and clamp to image bounds.
            x_min, x_max = sorted((x_min, x_max))
            y_min, y_max = sorted((y_min, y_max))
            x_min = max(0.0, min(float(IMG_W), x_min))
            x_max = max(0.0, min(float(IMG_W), x_max))
            y_min = max(0.0, min(float(IMG_H), y_min))
            y_max = max(0.0, min(float(IMG_H), y_max))

            bw = x_max - x_min
            bh = y_max - y_min
            # Drop degenerate boxes (zero/negative size after clamping).
            if bw <= 0.0 or bh <= 0.0:
                stats["boxes_skipped_invalid"] += 1
                continue

            cx = (x_min + x_max) / 2.0 / IMG_W
            cy = (y_min + y_max) / 2.0 / IMG_H
            nw = bw / IMG_W
            nh = bh / IMG_H

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats["boxes_kept"] += 1
            stats["class_counts"][CLASS_NAMES[cls_id]] += 1

        # Write a .txt next to the image (empty file = background sample).
        label_path = os.path.splitext(img_path)[0] + ".txt"
        with open(label_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(lines))
        stats["labels_written"] += 1

    return stats


def print_stats(name: str, stats: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  images seen        : {stats['images']}")
    print(f"  images missing     : {stats['images_missing']}")
    print(f"  label files written: {stats['labels_written']}")
    print(f"  boxes total        : {stats['boxes_total']}")
    print(f"  boxes kept         : {stats['boxes_kept']}")
    print(f"  boxes invalid      : {stats['boxes_skipped_invalid']}")
    print(f"  boxes unknown label: {stats['boxes_skipped_unknown']}")
    print(f"  class counts       : {dict(stats['class_counts'])}")
    if stats["unknown_labels"]:
        print(f"  unknown labels     : {dict(stats['unknown_labels'])}")


def main():
    parser = argparse.ArgumentParser(description="Convert BSTLD YAML to YOLO labels.")
    parser.add_argument(
        "--yaml",
        type=str,
        required=True,
        help="Path to a BSTLD annotation yaml (train.yaml or test.yaml).",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        action="append",
        default=[],
        help="Extra directory to search recursively for PNG images (repeatable). "
        "Use this to point at sibling folders where image zip parts were "
        "extracted, e.g. --images-root datasets",
    )
    args = parser.parse_args()

    if not os.path.exists(args.yaml):
        print(f"Error: yaml not found at {args.yaml}")
        sys.exit(1)

    stats = convert_one(args.yaml, image_roots=args.images_root)
    print_stats(os.path.basename(args.yaml), stats)
    print("\nDone.")


if __name__ == "__main__":
    main()
