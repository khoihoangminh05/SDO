"""Colab zip extraction helpers."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path


def list_zip_preview(zip_path: str, limit: int = 15) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    print(f"Zip has {len(names)} entries. First {limit}:")
    for name in names[:limit]:
        print(f"  {name}")
    return names


def setup_sdo_repo(content_root: str = "/content", drive_dir: str | None = None) -> tuple[str, str]:
    """
    Extract ttld_colab_code.zip and ensure /content/SDO/ttld-net exists.

    Returns (repo_path, ttld_path).
    """
    repo = os.path.join(content_root, "SDO")
    ttld = os.path.join(repo, "ttld-net")

    if os.path.isdir(ttld) and os.path.isfile(os.path.join(ttld, "train.py")):
        print("Code already at", ttld)
        return repo, ttld

    if drive_dir is None:
        drive_dir = "/content/drive/MyDrive/SDO_train"
    code_zip = os.path.join(drive_dir, "ttld_colab_code.zip")
    if not os.path.isfile(code_zip):
        raise FileNotFoundError(f"Missing {code_zip} — upload ttld_colab_code.zip to Drive")

    list_zip_preview(code_zip)
    with zipfile.ZipFile(code_zip) as zf:
        zf.extractall(content_root)

    # Normalize common zip layouts
    candidates: list[str] = []
    for base, dirs, files in os.walk(content_root):
        if base.endswith("ttld-net") and "train.py" in files:
            candidates.append(base)

    if not candidates:
        raise RuntimeError(
            "Không tìm thấy ttld-net/train.py sau giải nén. "
            "Tạo lại zip bằng pack_colab.ps1 và upload lại."
        )

    src_ttld = candidates[0]
    os.makedirs(repo, exist_ok=True)

    if os.path.abspath(src_ttld) != os.path.abspath(ttld):
        if os.path.exists(ttld):
            shutil.rmtree(ttld)
        shutil.move(src_ttld, ttld)
        print("Moved ttld-net ->", ttld)

    # apps/worker may be at /content/apps or /content/SDO/apps
    apps_dst = os.path.join(repo, "apps")
    if not os.path.isdir(apps_dst):
        for apps_src in [os.path.join(content_root, "apps"), os.path.join(repo, "apps")]:
            if os.path.isdir(apps_src) and os.path.abspath(apps_src) != os.path.abspath(apps_dst):
                shutil.move(apps_src, apps_dst)
                print("Moved apps ->", apps_dst)
                break

    yolo = os.path.join(repo, "apps/worker/models/yolo26_p2.yaml")
    if not os.path.isfile(yolo):
        raise RuntimeError(f"Missing {yolo} — zip thiếu yolo26_p2.yaml, tạo lại zip")

    print("REPO OK:", repo)
    print("TTLD OK:", ttld)
    print("yolo yaml:", yolo)
    return repo, ttld
