#!/usr/bin/env python3
"""Verify PyTorch can call torch.from_numpy (required by Ultralytics)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.preflight import verify_numpy_torch, verify_polars, verify_ultralytics_runtime


def main() -> int:
    try:
        verify_ultralytics_runtime()
        return 0
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
