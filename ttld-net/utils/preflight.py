"""Preflight checks for PyTorch + NumPy compatibility."""

from __future__ import annotations


def verify_numpy_torch() -> None:
    """
    Ultralytics calls torch.from_numpy() in preprocess — fails if NumPy ABI mismatches PyTorch.

    Common fix on conda GPU envs:
      pip install "numpy>=1.23.5,<2.0" --force-reinstall
    """
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency: {exc}") from exc

    if not hasattr(torch, "from_numpy"):
        raise RuntimeError("PyTorch installation is broken (no from_numpy)")

    try:
        arr = np.zeros((2, 3), dtype=np.uint8)
        tensor = torch.from_numpy(arr)
        _ = tensor.shape
    except RuntimeError as exc:
        if "Numpy is not available" in str(exc):
            raise RuntimeError(
                "PyTorch cannot use NumPy (ABI mismatch). On GPU server run:\n"
                '  pip install "numpy>=1.23.5,<2.0" --force-reinstall\n'
                "  python scripts/check_numpy_torch.py\n"
                "Then retry training."
            ) from exc
        raise

    print(f"OK numpy={np.__version__} torch={torch.__version__} from_numpy works")


def verify_polars() -> None:
    """
    Ultralytics imports polars — default wheel needs AVX2; older CPUs crash with SIGILL.

    Fix: pip install 'polars[rtcompat]>=0.20.0' --force-reinstall
    """
    try:
        import polars as pl
    except ImportError:
        print("WARN polars not installed (ultralytics may install it)")
        return

    try:
        pl.DataFrame({"a": [1, 2, 3]}).height
    except Exception as exc:
        raise RuntimeError(
            "Polars crashed on this CPU. Run:\n"
            "  pip uninstall polars polars-runtime-32 -y\n"
            "  pip install 'polars[rtcompat]>=0.20.0'\n"
            "Then retry training."
        ) from exc

    print(f"OK polars={pl.__version__}")


def verify_ultralytics_runtime() -> None:
    """Run all preflight checks before YOLO train/val."""
    verify_numpy_torch()
    verify_polars()
