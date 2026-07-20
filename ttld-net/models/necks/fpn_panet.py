"""FPN + PANet neck and feature projection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjection(nn.Module):
    """Project feature maps to a shared embedding dimension (default 256)."""

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FPNPANet(nn.Module):
    """
    Feature Pyramid Network (top-down) + PANet (bottom-up).

    All outputs are projected to ``out_channels`` (default 256) so TinyGenerator
    and context branch share a unified embedding width.
    """

    def __init__(
        self,
        in_channels: tuple[int, ...] = (16, 64, 128, 128, 256),
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        if len(in_channels) != 5:
            raise ValueError(f"Expected 5 pyramid levels, got {len(in_channels)}")
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lateral = nn.ModuleList(
            [nn.Conv2d(ch, out_channels, kernel_size=1) for ch in in_channels]
        )
        self.smooth_td = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in range(4)]
        )
        self.down = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                for _ in range(4)
            ]
        )
        self.smooth_bu = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in range(4)]
        )

    def forward(
        self, features: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != 5:
            raise ValueError(f"Expected 5 feature maps, got {len(features)}")

        laterals = [lat(feat) for lat, feat in zip(self.lateral, features)]

        # Top-down: P5 → P4 → P3 → P2 → P1
        td: list[torch.Tensor] = [laterals[4]]
        for idx in range(3, -1, -1):
            up = F.interpolate(td[0], size=laterals[idx].shape[-2:], mode="nearest")
            fused = laterals[idx] + up
            smooth = self.smooth_td[3 - idx](fused)
            td.insert(0, smooth)

        # Bottom-up: P1 → P2 → P3 → P4 → P5
        bu: list[torch.Tensor] = [td[0]]
        for idx in range(4):
            down = self.down[idx](bu[-1])
            if down.shape[-2:] != td[idx + 1].shape[-2:]:
                down = F.interpolate(down, size=td[idx + 1].shape[-2:], mode="nearest")
            fused = td[idx + 1] + down
            bu.append(self.smooth_bu[idx](fused))

        return bu[0], bu[1], bu[2], bu[3], bu[4]

    def project_context(
        self, p4: torch.Tensor, p5: torch.Tensor, c4: int, c5: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project deep features for the semantic context branch (Phase 3)."""
        proj_p4 = FeatureProjection(c4, 256).to(p4.device)
        proj_p5 = FeatureProjection(c5, 256).to(p5.device)
        return proj_p4(p4), proj_p5(p5)
