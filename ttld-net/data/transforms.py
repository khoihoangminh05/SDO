"""Augmentation pipeline for Bosch dataset."""

from __future__ import annotations

from typing import Any

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# BSTLD native resolution: 1280 x 720 (W x H)
TARGET_HEIGHT = 720
TARGET_WIDTH = 1280


class ComposeWithBoxes:
    """Apply transforms that may update bounding boxes."""

    def __init__(self, transforms_list: list[Any]) -> None:
        self.transforms_list = transforms_list

    def __call__(self, image: Any, boxes: torch.Tensor) -> tuple[Any, torch.Tensor]:
        for transform in self.transforms_list:
            image, boxes = transform(image, boxes)
        return image, boxes


class ResizeWithBoxes:
    """Resize image and scale box coordinates to target H x W."""

    def __init__(self, height: int = TARGET_HEIGHT, width: int = TARGET_WIDTH) -> None:
        self.height = height
        self.width = width

    def __call__(self, image: Any, boxes: torch.Tensor) -> tuple[Any, torch.Tensor]:
        orig_w, orig_h = image.size
        image = TF.resize(image, [self.height, self.width])
        if boxes.numel() == 0:
            return image, boxes
        scale_x = self.width / orig_w
        scale_y = self.height / orig_h
        boxes = boxes.clone()
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y
        return image, boxes


class RandomHorizontalFlip:
    """Flip image and boxes horizontally."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Any, boxes: torch.Tensor) -> tuple[Any, torch.Tensor]:
        if torch.rand(1).item() >= self.p:
            return image, boxes
        image = TF.hflip(image)
        if boxes.numel() == 0:
            return image, boxes
        width = image.size[0] if hasattr(image, "size") else image.shape[-1]
        boxes = boxes.clone()
        boxes[:, 0] = width - boxes[:, 0]
        return image, boxes


class ColorJitterWrapper:
    """Color jitter without box changes."""

    def __init__(self, **kwargs: float) -> None:
        self.jitter = transforms.ColorJitter(**kwargs)

    def __call__(self, image: Any, boxes: torch.Tensor) -> tuple[Any, torch.Tensor]:
        return self.jitter(image), boxes


class ToTensorNormalize:
    """Convert PIL image to normalized tensor."""

    def __call__(self, image: Any, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, boxes


def get_train_transforms(height: int = TARGET_HEIGHT, width: int = TARGET_WIDTH) -> ComposeWithBoxes:
    """Default training augmentation chain (Phase 1)."""
    return ComposeWithBoxes(
        [
            ResizeWithBoxes(height=height, width=width),
            RandomHorizontalFlip(p=0.5),
            ColorJitterWrapper(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            ToTensorNormalize(),
        ]
    )


def get_val_transforms(height: int = TARGET_HEIGHT, width: int = TARGET_WIDTH) -> ComposeWithBoxes:
    """Validation transforms without random augmentation."""
    return ComposeWithBoxes([ResizeWithBoxes(height=height, width=width), ToTensorNormalize()])
