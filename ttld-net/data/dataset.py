"""Bosch Small Traffic Lights Dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

CLASS_MAPPING: dict[str, int] = {
    "Green": 0,
    "GreenLeft": 0,
    "GreenRight": 0,
    "GreenStraight": 0,
    "Yellow": 1,
    "Red": 2,
    "RedLeft": 2,
    "RedRight": 2,
    "RedStraight": 2,
    "off": 3,
}

CLASS_NAMES: list[str] = ["Green", "Yellow", "Red", "Off"]


class BoschDataset(Dataset):
    """Dataset for Bosch YAML labels with 4-class TTLD mapping."""

    def __init__(
        self,
        yaml_path: str | Path,
        transform: Callable[..., Any] | None = None,
        images_root: str | Path | None = None,
    ) -> None:
        self.yaml_path = Path(yaml_path)
        self.transform = transform
        self.images_root = Path(images_root) if images_root else self.yaml_path.parent
        self.samples = self._load_yaml(self.yaml_path)

    def _load_yaml(self, yaml_path: Path) -> list[dict[str, Any]]:
        with yaml_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        samples: list[dict[str, Any]] = []

        # BSTLD format: list of {path, boxes}
        if isinstance(data, list):
            for entry in data:
                rel = entry.get("path", "")
                image_path = self._resolve_image_path(rel)
                samples.append({"image_path": image_path, "boxes": entry.get("boxes") or []})
            return samples

        # Alternate dict format: {image_path: {boxes: [...]}}
        for image_key, payload in data.items():
            image_path = self._resolve_image_path(image_key)
            boxes = payload.get("boxes", []) if isinstance(payload, dict) else []
            samples.append({"image_path": image_path, "boxes": boxes})
        return samples

    def _resolve_image_path(self, image_key: str) -> Path:
        path = Path(image_key)
        if path.is_file():
            return path
        # BSTLD paths like ./rgb/train/bag_name/frame.png
        candidate = (self.images_root / image_key).resolve()
        if candidate.is_file():
            return candidate
        return (self.images_root / Path(image_key).name).resolve()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        sample = self.samples[idx]
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except OSError as exc:
            raise RuntimeError(f"Corrupt image: {sample['image_path']}") from exc

        boxes = self._parse_boxes(sample["boxes"])
        if self.transform is not None:
            image, boxes = self.transform(image, boxes)
        elif not isinstance(image, torch.Tensor):
            image = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0

        return image, boxes

    def _parse_boxes(self, raw_boxes: list[dict[str, Any]]) -> torch.Tensor:
        parsed: list[list[float]] = []
        for box in raw_boxes:
            label = box.get("label", "off")
            cls_id = CLASS_MAPPING.get(label, 3)

            if "x_center" in box:
                parsed.append(
                    [
                        float(box["x_center"]),
                        float(box["y_center"]),
                        float(box["w"]),
                        float(box["h"]),
                        float(cls_id),
                    ]
                )
            elif "x_min" in box:
                x_min, x_max = float(box["x_min"]), float(box["x_max"])
                y_min, y_max = float(box["y_min"]), float(box["y_max"])
                w, h = x_max - x_min, y_max - y_min
                parsed.append([x_min + w / 2, y_min + h / 2, w, h, float(cls_id)])

        if not parsed:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(parsed, dtype=torch.float32)


def custom_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Collate images and variable-length box tensors."""
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def create_dataloader(
    yaml_path: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    transform: Callable[..., Any] | None = None,
    images_root: str | Path | None = None,
) -> DataLoader:
    """Factory for Bosch DataLoader with custom collate."""
    yaml_path = Path(yaml_path)
    root = Path(images_root) if images_root else yaml_path.parent
    dataset = BoschDataset(yaml_path, transform=transform, images_root=root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate,
    )
