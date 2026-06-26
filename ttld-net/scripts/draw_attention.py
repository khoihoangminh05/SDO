#!/usr/bin/env python3
"""Visualize deformable attention offsets on validation images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m4_full_ttld.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="logs/attention_viz.png")
    args = parser.parse_args()

    raise NotImplementedError(
        f"Phase 7 (T7.4): visualize attention for {args.image} → {args.output}"
    )


if __name__ == "__main__":
    main()
