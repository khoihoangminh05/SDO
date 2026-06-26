#!/usr/bin/env python3
"""Exploratory data analysis for Bosch BSTLD dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def analyze_bbox_distribution(yaml_path: Path, output_dir: Path) -> None:
    """
    Analyze bbox size distribution, class balance, and tiny-object ratio.

    Phase 1 (T1.1): implement histograms and console statistics.
    """
    raise NotImplementedError(f"Phase 1: implement EDA for {yaml_path} → {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, help="Bosch train.yaml path")
    parser.add_argument("--output", default="logs", help="Directory for plots")
    args = parser.parse_args()
    analyze_bbox_distribution(Path(args.yaml), Path(args.output))


if __name__ == "__main__":
    main()
