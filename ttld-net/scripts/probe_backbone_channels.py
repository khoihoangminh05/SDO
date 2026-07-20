#!/usr/bin/env python3
"""T3.2 / Phase 2 helper — probe YOLO26 backbone channel dims."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe YOLO26 P1–P5 channels")
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--yaml",
        default=None,
        help="Path to yolo26_p2.yaml (default: apps/worker/models/yolo26_p2.yaml)",
    )
    parser.add_argument(
        "--output",
        default="logs/backbone_channels.txt",
        help="Where to write channel dump",
    )
    args = parser.parse_args()

    from models.backbones.yolo26 import YOLO26Backbone

    yaml_path = Path(args.yaml) if args.yaml else (
        ROOT.parent / "apps" / "worker" / "models" / "yolo26_p2.yaml"
    )
    backbone = YOLO26Backbone(yaml_path=yaml_path)
    dims = backbone.channel_dims((args.height, args.width))

    lines = [
        f"input: (1, 3, {args.height}, {args.width})",
        f"yaml: {yaml_path}",
        *[f"{k.upper()} channels = {v}" for k, v in dims.items()],
        f"C4 = {dims['p4']}",
        f"C5 = {dims['p5']}",
    ]
    text = "\n".join(lines) + "\n"
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
