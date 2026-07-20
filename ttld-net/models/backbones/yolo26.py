"""YOLO26 backbone exposing P1–P5 feature maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Backbone layer indices in apps/worker/models/yolo26_p2.yaml
_BACKBONE_LAYERS: dict[str, int] = {
    "p1": 0,   # stride 2
    "p2": 2,   # stride 4
    "p3": 4,   # stride 8
    "p4": 6,   # stride 16
    "p5": 10,  # stride 32 (after C2PSA)
}


class YOLO26Backbone(nn.Module):
    """
    YOLO26 / YOLO26-P2 backbone wrapper.

    Runs Ultralytics DetectionModel layers and returns raw backbone pyramids
    (P1–P5) before the Detect head. Neck fusion is handled by FPNPANet.
    """

    def __init__(
        self,
        yaml_path: str | Path | None = None,
        weights: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.yaml_path = Path(yaml_path) if yaml_path else (
            Path(__file__).resolve().parents[3]
            / "apps"
            / "worker"
            / "models"
            / "yolo26_p2.yaml"
        )
        self.weights = Path(weights) if weights else None
        self._impl: nn.Module | None = None
        self._channel_dims: dict[str, int] = {}
        self._save_indices = set(_BACKBONE_LAYERS.values())

    def _lazy_init(self) -> None:
        if self._impl is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise NotImplementedError(
                "YOLO26Backbone requires Ultralytics. Install ultralytics>=8.0."
            ) from exc

        if not self.yaml_path.is_file():
            raise FileNotFoundError(f"YOLO yaml not found: {self.yaml_path}")

        yolo = YOLO(str(self.yaml_path))
        if self.weights is not None and self.weights.is_file():
            yolo = YOLO(str(self.weights))
        self._impl = yolo.model
        self._impl.eval()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Return (P1, P2, P3, P4, P5).

        Prefer input H,W divisible by 32 (e.g. 736×1280). Raw 720×1280 can
        break Concat in the YOLO neck; backbone-only path still works for
        most sizes but tests use 640 or 736.
        """
        self._lazy_init()
        assert self._impl is not None
        layers = self._impl.model
        save = set(getattr(self._impl, "save", [])) | self._save_indices

        y: list[Any] = []
        collected: dict[int, torch.Tensor] = {}
        out = x

        for i, layer in enumerate(layers):
            # Stop before Detect head — we only need backbone pyramids.
            if layer.__class__.__name__ == "Detect":
                break

            f = layer.f
            if f != -1:
                out = y[f] if isinstance(f, int) else [out if j == -1 else y[j] for j in f]
            out = layer(out)
            y.append(out if (getattr(layer, "i", i) in save or i in self._save_indices) else None)

            if i in self._save_indices and isinstance(out, torch.Tensor):
                collected[i] = out

            # Backbone ends at layer 10 (C2PSA); neck starts at upsample 11.
            if i >= max(self._save_indices):
                break

        missing = [name for name, idx in _BACKBONE_LAYERS.items() if idx not in collected]
        if missing:
            raise RuntimeError(f"Failed to collect backbone features: {missing}")

        return tuple(collected[_BACKBONE_LAYERS[k]] for k in ("p1", "p2", "p3", "p4", "p5"))

    @torch.no_grad()
    def channel_dims(self, image_size: tuple[int, int] = (640, 640)) -> dict[str, int]:
        """Probe channel sizes for each pyramid level."""
        if self._channel_dims:
            return self._channel_dims

        height, width = image_size
        dummy = torch.zeros(1, 3, height, width)
        device = next(self.parameters(), torch.tensor(0)).device
        # Module may have no parameters until lazy init
        self._lazy_init()
        assert self._impl is not None
        device = next(self._impl.parameters()).device
        dummy = dummy.to(device)
        outputs = self.forward(dummy)
        names = ["p1", "p2", "p3", "p4", "p5"]
        self._channel_dims = {name: int(tensor.shape[1]) for name, tensor in zip(names, outputs)}
        return self._channel_dims
