#!/usr/bin/env python3
"""Phase 0 environment verification."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_checks(strict_gpu: bool, dev_mode: bool) -> tuple[list[tuple[str, bool, str]], bool]:
    root = Path(__file__).resolve().parent
    repo = root.parent
    checks: list[tuple[str, bool, str]] = []
    required_fail = False

    # T0.1 — PyTorch + CUDA
    try:
        import torch

        checks.append(("PyTorch", True, torch.__version__))
        cuda_ok = torch.cuda.is_available()
        cuda_detail = torch.version.cuda or "CPU only"
        if cuda_ok and torch.cuda.device_count() > 0:
            cuda_detail += f" | {torch.cuda.get_device_name(0)}"
        checks.append(("CUDA available", cuda_ok, cuda_detail))
    except ImportError:
        checks.append(("PyTorch", False, "not installed"))
        required_fail = True

    # T0.3 — MMCV deformable attention
    try:
        from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401

        checks.append(("MMCV DeformableAttention", True, "OK"))
    except ImportError as exc:
        detail = "not installed"
        if not dev_mode:
            helper = root / "scripts" / "install_mmcv.py"
            detail = f"run: python {helper.name}  (GPU server Linux + CUDA required)"
        checks.append(("MMCV DeformableAttention", False, detail))
        if strict_gpu:
            required_fail = True

    # T0.2 — Core deps
    for name, module in [
        ("yaml", "yaml"),
        ("cv2", "cv2"),
        ("tensorboard", "tensorboard"),
        ("ultralytics", "ultralytics"),
        ("pytest", "pytest"),
    ]:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "OK")
            checks.append((name, True, str(version)))
        except ImportError:
            checks.append((name, False, "not installed"))
            required_fail = True

    # T0.5 — Dataset files
    train_yaml = repo / "apps/worker/datasets/dataset_train_rgb/train.yaml"
    val_images = repo / "apps/worker/datasets/dataset_val_sample/rgb/val"
    bstld_yaml = repo / "apps/worker/datasets/bstld.yaml"
    checks.append(("BSTLD train.yaml", train_yaml.is_file(), str(train_yaml)))
    val_ok = val_images.is_dir() and any(val_images.glob("*.png"))
    checks.append(("BSTLD val split", val_ok, str(val_images)))
    checks.append(("BSTLD ultralytics yaml", bstld_yaml.is_file(), str(bstld_yaml)))
    if not train_yaml.is_file() or not bstld_yaml.is_file():
        required_fail = True

    # Run verify_dataset script
    verify_script = root / "scripts/verify_dataset.py"
    if verify_script.is_file():
        result = subprocess.run(
            [sys.executable, str(verify_script)],
            capture_output=True,
            text=True,
            cwd=str(root),
        )
        dataset_ok = result.returncode == 0
        detail = "PASS" if dataset_ok else "FAIL — run scripts/verify_dataset.py"
        checks.append(("Dataset verification (T0.5)", dataset_ok, detail))
        if not dataset_ok:
            required_fail = True

    if strict_gpu:
        cuda_check = next((c for c in checks if c[0] == "CUDA available"), None)
        if cuda_check and not cuda_check[1]:
            required_fail = True

    if dev_mode:
        # Dev machine: MMCV/CUDA warnings only
        for label in ("CUDA available", "MMCV DeformableAttention"):
            checks = [(l, o, d) for l, o, d in checks]

    return checks, required_fail and not dev_mode


def main() -> int:
    parser = argparse.ArgumentParser(description="TTLD-Net Phase 0 environment gate")
    parser.add_argument(
        "--strict-gpu",
        action="store_true",
        help="Require CUDA + MMCV (GPU server)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: pass without CUDA/MMCV (Windows dev machine)",
    )
    parser.add_argument(
        "--log",
        default="logs/env_check.txt",
        help="Write results to this file",
    )
    args = parser.parse_args()

    checks, failed = _run_checks(strict_gpu=args.strict_gpu, dev_mode=args.dev)

    lines = ["TTLD-Net environment check", "=" * 40]
    for label, ok, detail in checks:
        mark = "OK" if ok else "FAIL"
        lines.append(f"[{mark}] {label}: {detail}")

    if args.dev:
        lines.append("")
        lines.append("Mode: DEV — CUDA/MMCV optional (run setup_phase0.sh on GPU server)")
    elif args.strict_gpu:
        lines.append("")
        lines.append("Mode: STRICT GPU — all checks required")

    cuda_ok = any(c[0] == "CUDA available" and c[1] for c in checks)
    if not cuda_ok:
        lines.append("")
        lines.append("Note: CUDA not available on this machine — train on GPU server.")

    output = "\n".join(lines)
    print(output)

    log_path = Path(__file__).resolve().parent / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output + "\n", encoding="utf-8")
    print(f"\nLog: {log_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
