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

# Ultralytics BSTLD labels: red=0, yellow=1, green=2, off=3
YOLO_TO_TTLD_CLASS: dict[int, int] = {0: 2, 1: 1, 2: 0, 3: 3}


def resolve_dataset_path(path: str | Path) -> tuple[Path, str]:
    """
    Resolve train/val source to either Bosch YAML or YOLO image directory.

    Returns:
        (resolved_path, mode) where mode is ``yaml`` or ``yolo_dir``.
    """
    from utils.config import resolve_path

    candidate = resolve_path(str(path))
    if candidate.is_file():
        return candidate, "yaml"
    if candidate.is_dir():
        return candidate, "yolo_dir"

    # Common layout: val.yaml missing but rgb/val/*.png exists (sample_val.py)
    fallbacks = [
        candidate.parent / "rgb" / "val",
        candidate.parent / "rgb" / "train",
        candidate.parent.parent / "rgb" / "val",
    ]
    for fallback in fallbacks:
        if fallback.is_dir() and any(fallback.glob("**/*.png")):
            return fallback.resolve(), "yolo_dir"

    raise FileNotFoundError(
        f"Dataset not found: {candidate}. "
        "Expected Bosch YAML file or directory of PNG+TXT (YOLO format). "
        "Val split often lives at apps/worker/datasets/dataset_val_sample/rgb/val"
    )


class YoloDirDataset(Dataset):
    """Load BSTLD images with sidecar YOLO .txt labels (val sample layout)."""

    def __init__(
        self,
        image_dir: str | Path,
        transform: Callable[..., Any] | None = None,
        image_width: int = 1280,
        image_height: int = 720,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_width = image_width
        self.image_height = image_height
        self.samples = sorted(self.image_dir.glob("**/*.png"))
        if not self.samples:
            raise FileNotFoundError(f"No PNG images under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        image_path = self.samples[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError as exc:
            raise RuntimeError(f"Corrupt image: {image_path}") from exc

        boxes = self._load_yolo_labels(image_path.with_suffix(".txt"))
        if self.transform is not None:
            image, boxes = self.transform(image, boxes)
        elif not isinstance(image, torch.Tensor):
            image = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0

        return image, boxes

    def _load_yolo_labels(self, label_path: Path) -> torch.Tensor:
        if not label_path.is_file():
            return torch.zeros((0, 5), dtype=torch.float32)

        parsed: list[list[float]] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            yolo_cls = int(float(parts[0]))
            cx = float(parts[1]) * self.image_width
            cy = float(parts[2]) * self.image_height
            bw = float(parts[3]) * self.image_width
            bh = float(parts[4]) * self.image_height
            cls_id = YOLO_TO_TTLD_CLASS.get(yolo_cls, 3)
            parsed.append([cx, cy, bw, bh, float(cls_id)])

        if not parsed:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(parsed, dtype=torch.float32)


class BoschDataset(Dataset):
    """Dataset for Bosch YAML labels with 4-class TTLD mapping."""

    def __init__(
        self,
        yaml_path: str | Path,
        transform: Callable[..., Any] | None = None,
        images_root: str | Path | None = None,
        skip_missing: bool = True,
    ) -> None:
        self.yaml_path = Path(yaml_path)
        self.transform = transform
        self.images_root = Path(images_root) if images_root else self.yaml_path.parent
        samples = self._load_yaml(self.yaml_path)
        if skip_missing:
            samples = [s for s in samples if s["image_path"].is_file()]
        self.samples = samples

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


class SubsetDataset(Dataset):
    """Use the first fraction of a dataset (deterministic, fast ablation)."""

    def __init__(self, base: Dataset, ratio: float = 1.0) -> None:
        if not 0 < ratio <= 1.0:
            raise ValueError(f"train_subset_ratio must be in (0, 1], got {ratio}")
        self.base = base
        n = len(base)
        self.indices = list(range(max(1, int(n * ratio))))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.base[self.indices[idx]]


def create_dataloader(
    yaml_path: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    transform: Callable[..., Any] | None = None,
    images_root: str | Path | None = None,
    split: str = "train",
    image_size: tuple[int, int] | None = None,
    subset_ratio: float = 1.0,
) -> DataLoader:
    """Factory for Bosch YAML or YOLO-directory DataLoader with custom collate."""
    from data.transforms import get_train_transforms, get_val_transforms

    resolved, mode = resolve_dataset_path(yaml_path)
    height, width = image_size if image_size else (720, 1280)
    if transform is None:
        transform = (
            get_train_transforms(height=height, width=width)
            if split == "train"
            else get_val_transforms(height=height, width=width)
        )

    if mode == "yaml":
        root = Path(images_root) if images_root else resolved.parent
        dataset: Dataset = BoschDataset(resolved, transform=transform, images_root=root)
    else:
        dataset = YoloDirDataset(
            resolved,
            transform=transform,
            image_width=width,
            image_height=height,
        )

    if split == "train" and subset_ratio < 1.0:
        dataset = SubsetDataset(dataset, ratio=subset_ratio)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate,
    )
