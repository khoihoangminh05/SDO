"""
Prune BSTLD YOLO datasets to stay under a size budget.

Usage (from apps/worker/):
    python scripts/prune_datasets.py --target-gb 18 --dry-run
    python scripts/prune_datasets.py --target-gb 18

Keeps train intact; reduces test + val with a fixed seed for reproducibility.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def list_pngs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def delete_samples(pngs: list[Path], dry_run: bool) -> int:
    removed = 0
    for png in pngs:
        txt = png.with_suffix(".txt")
        if dry_run:
            removed += 1
            continue
        png.unlink(missing_ok=True)
        txt.unlink(missing_ok=True)
        removed += 1
    return removed


def resample_val(src: Path, dst: Path, count: int, seed: int, dry_run: bool) -> int:
    all_pngs = list_pngs(src)
    random.seed(seed)
    picked = random.sample(all_pngs, min(count, len(all_pngs)))

    if dry_run:
        return len(picked)

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for png_path in picked:
        txt_path = png_path.with_suffix(".txt")
        dst_png = dst / png_path.name
        dst_txt = dst_png.with_suffix(".txt")
        shutil.copy2(png_path, dst_png)
        if txt_path.exists():
            shutil.copy2(txt_path, dst_txt)
        else:
            dst_txt.write_text("", encoding="utf-8")

    return len(picked)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gb", type=float, default=18.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-count", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    worker = Path(__file__).resolve().parents[1]
    datasets = worker / "datasets"
    train_dir = datasets / "dataset_train_rgb"
    test_dir = datasets / "dataset_test_rgb" / "rgb" / "test"
    val_dir = datasets / "dataset_val_sample" / "rgb" / "val"

    before = {
        "train": dir_size_gb(train_dir),
        "test": dir_size_gb(test_dir.parent.parent),
        "val": dir_size_gb(val_dir.parent.parent),
    }
    before_total = sum(before.values())

    print("=== Before ===")
    for k, v in before.items():
        print(f"  {k}: {v:.2f} GB")
    print(f"  total: {before_total:.2f} GB")
    print(f"  target: {args.target_gb:.2f} GB\n")

    train_gb = before["train"]
    val_target_gb = args.val_count * (before["test"] / max(len(list_pngs(test_dir)), 1))
    test_budget_gb = args.target_gb - train_gb - val_target_gb
    avg_img_gb = before["test"] / max(len(list_pngs(test_dir)), 1)
    test_keep = max(1000, int(test_budget_gb / avg_img_gb))

    test_pngs = list_pngs(test_dir)
    random.seed(args.seed)
    keep_set = set(random.sample(test_pngs, min(test_keep, len(test_pngs))))
    remove_pngs = [p for p in test_pngs if p not in keep_set]

    print(f"Plan (seed={args.seed}):")
    print(f"  train: keep all ({len(list_pngs(train_dir))} images)")
    print(f"  test:  keep {len(keep_set)}, remove {len(remove_pngs)}")
    print(f"  val:   resample {args.val_count} from test\n")

    if args.dry_run:
        est_test = len(keep_set) * avg_img_gb
        est_val = args.val_count * avg_img_gb
        est_total = train_gb + est_test + est_val
        print(f"Estimated after: {est_total:.2f} GB")
        return

    removed = delete_samples(remove_pngs, dry_run=False)
    print(f"Removed {removed} test samples")

    val_n = resample_val(test_dir, val_dir, args.val_count, args.seed, dry_run=False)
    print(f"Val resampled: {val_n} images")

    after = {
        "train": dir_size_gb(train_dir),
        "test": dir_size_gb(test_dir.parent.parent),
        "val": dir_size_gb(val_dir.parent.parent),
    }
    after_total = sum(after.values())

    print("\n=== After ===")
    for k, v in after.items():
        print(f"  {k}: {v:.2f} GB")
    print(f"  total: {after_total:.2f} GB")
    print(f"  saved: {before_total - after_total:.2f} GB")


if __name__ == "__main__":
    main()
