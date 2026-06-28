"""Shared Bosch YAML parsing for dataset and EDA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from data.dataset import CLASS_MAPPING, CLASS_NAMES


@dataclass
class BoxRecord:
    """Single annotation with pixel geometry."""

    label: str
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float


def parse_box(raw: dict[str, Any]) -> BoxRecord | None:
    """Convert a raw Bosch box dict to BoxRecord."""
    label = raw.get("label", "off")
    class_id = CLASS_MAPPING.get(label, 3)
    class_name = CLASS_NAMES[class_id]

    if "x_center" in raw:
        w, h = float(raw["w"]), float(raw["h"])
        return BoxRecord(
            label=label,
            class_id=class_id,
            class_name=class_name,
            x_center=float(raw["x_center"]),
            y_center=float(raw["y_center"]),
            width=w,
            height=h,
        )

    if "x_min" in raw:
        x_min, x_max = float(raw["x_min"]), float(raw["x_max"])
        y_min, y_max = float(raw["y_min"]), float(raw["y_max"])
        w, h = x_max - x_min, y_max - y_min
        if w <= 0 or h <= 0:
            return None
        return BoxRecord(
            label=label,
            class_id=class_id,
            class_name=class_name,
            x_center=x_min + w / 2,
            y_center=y_min + h / 2,
            width=w,
            height=h,
        )
    return None


def load_bosch_entries(yaml_path: Path, images_root: Path | None = None) -> list[dict[str, Any]]:
    """Load YAML entries with resolved image paths."""
    root = images_root or yaml_path.parent
    with yaml_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    entries: list[dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            rel = item.get("path", "")
            img_path = _resolve_image_path(root, rel)
            boxes = [parse_box(b) for b in (item.get("boxes") or [])]
            boxes = [b for b in boxes if b is not None]
            entries.append({"image_path": img_path, "boxes": boxes})
    else:
        for key, payload in data.items():
            img_path = _resolve_image_path(root, key)
            raw_boxes = payload.get("boxes", []) if isinstance(payload, dict) else []
            boxes = [parse_box(b) for b in raw_boxes]
            boxes = [b for b in boxes if b is not None]
            entries.append({"image_path": img_path, "boxes": boxes})
    return entries


def _resolve_image_path(root: Path, key: str) -> Path:
    path = Path(key)
    if path.is_file():
        return path.resolve()
    candidate = (root / key).resolve()
    if candidate.is_file():
        return candidate
    return (root / Path(key).name).resolve()
