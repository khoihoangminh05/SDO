"""YOLO26 backbone exposing P1–P5 feature maps."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class YOLO26Backbone(nn.Module):
    """
    YOLO26 backbone wrapper.

  Phase 2 will hook Ultralytics internals or a custom nn.Module stack to return
  all five pyramid levels. For now this module validates the forward contract.
    """

    def __init__(self, yaml_path: str | Path | None = None) -> None:
        super().__init__()
        self.yaml_path = yaml_path or (
            Path(__file__).resolve().parents[3]
            / "apps"
            / "worker"
            / "models"
            / "yolo26_p2.yaml"
        )
        self._impl: nn.Module | None = None
        self._channel_dims: dict[str, int] = {}

    def _lazy_init(self) -> None:
        if self._impl is not None:
            return
        try:
            from ultralytics import YOLO

            model = YOLO(str(self.yaml_path))
            self._impl = model.model
        except Exception as exc:  # pragma: no cover - optional ultralytics
            raise NotImplementedError(
                "YOLO26Backbone requires Ultralytics and a valid yolo26 yaml. "
                "Implement custom backbone layers in Phase 2."
            ) from exc

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Return (P1, P2, P3, P4, P5).

        Shapes for input (B, 3, 720, 1280):
          P1: (B, C1, 360, 640)
          P2: (B, C2, 180, 320)
          P3: (B, C3, 90, 160)
          P4: (B, C4, 45, 80)
          P5: (B, C5, 23, 40)
        """
        self._lazy_init()
        raise NotImplementedError(
            "Phase 2 (T2.1): override forward to emit P1–P5 from YOLO26 backbone."
        )

    def channel_dims(self) -> dict[str, int]:
        """Return channel sizes for each pyramid level after probing."""
        if not self._channel_dims:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 720, 1280)
                try:
                    outputs = self.forward(dummy)
                    names = ["p1", "p2", "p3", "p4", "p5"]
                    self._channel_dims = {
                        name: tensor.shape[1] for name, tensor in zip(names, outputs)
                    }
                except NotImplementedError:
                    self._channel_dims = {"p4": 512, "p5": 1024}
        return self._channel_dims
