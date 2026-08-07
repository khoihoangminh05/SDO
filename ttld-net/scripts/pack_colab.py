#!/usr/bin/env python3
"""Create ttld_colab_code.zip with guaranteed /content/SDO/ layout on Colab."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def pack(repo_root: Path, out_zip: Path) -> None:
    ttld = repo_root / "ttld-net"
    yolo_yaml = repo_root / "apps" / "worker" / "models" / "yolo26_p2.yaml"
    if not ttld.is_dir():
        raise FileNotFoundError(f"Missing {ttld}")
    if not yolo_yaml.is_file():
        raise FileNotFoundError(f"Missing {yolo_yaml}")

    skip_dirs = {".git", "__pycache__", "logs", "checkpoints", "results", "runs", ".pytest_cache"}
    skip_suffixes = {".pth", ".pt"}

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.is_file():
        out_zip.unlink()

    bstld_placeholder = (
        "path: PLACEHOLDER\n"
        "train: dataset_train_rgb/rgb/train\n"
        "val: dataset_val_sample/rgb/val\n"
        "nc: 4\n"
        "names:\n"
        "  0: red\n"
        "  1: yellow\n"
        "  2: green\n"
        "  3: off\n"
    )

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in ttld.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(ttld).as_posix()
            if any(part in skip_dirs for part in path.parts):
                continue
            if path.suffix.lower() in skip_suffixes:
                continue
            arc = f"SDO/ttld-net/{rel}"
            zf.write(path, arc)

        zf.write(yolo_yaml, "SDO/apps/worker/models/yolo26_p2.yaml")
        zf.writestr("SDO/apps/worker/datasets/bstld.yaml", bstld_placeholder)

    mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Created {out_zip} ({mb:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="SDO monorepo root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "ttld_colab_code.zip",
    )
    args = parser.parse_args()
    pack(args.repo.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
