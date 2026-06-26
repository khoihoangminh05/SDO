"""FPN + PANet neck and feature projection."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """Project feature maps to a shared embedding dimension (default 256)."""

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FPNPANet(nn.Module):
    """
    Feature Pyramid Network + PANet neck.

    Phase 2 (T2.2) will wire top-down and bottom-up pathways.
    """

    def __init__(self, in_channels: tuple[int, ...] = (256, 512, 512, 1024, 1024)) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.lateral = nn.ModuleList(
            [nn.Conv2d(ch, 256, kernel_size=1) for ch in in_channels]
        )

    def forward(
        self, features: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Phase 2 (T2.2): implement FPN + PANet forward pass.")

    def project_context(
        self, p4: torch.Tensor, p5: torch.Tensor, c4: int, c5: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project deep features for the semantic context branch (Phase 3)."""
        proj_p4 = FeatureProjection(c4, 256)
        proj_p5 = FeatureProjection(c5, 256)
        return proj_p4(p4), proj_p5(p5)
