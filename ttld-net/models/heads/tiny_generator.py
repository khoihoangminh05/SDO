"""High-recall candidate generator (Stage 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from models.heads.soft_nms import soft_nms


@dataclass
class Candidate:
    """Single traffic-light candidate."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    is_false_positive: bool = False


class TinyGenerator(nn.Module):
    """
    High-recall candidate generator on shallow features P1, P2, P3.

    Target: Recall > 95% with conf_threshold=0.05 and Soft-NMS.
    """

    def __init__(
        self,
        conf_threshold: float = 0.05,
        soft_nms_sigma: float = 0.5,
        fcand_dim: int = 256,
    ) -> None:
        super().__init__()
        self.conf_threshold = conf_threshold
        self.soft_nms_sigma = soft_nms_sigma
        self.fcand_dim = fcand_dim
        self.feature_proj = nn.Linear(fcand_dim, fcand_dim)

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
    ) -> tuple[list[list[dict[str, Any]]], torch.Tensor]:
        """
        Args:
            p1, p2, p3: Shallow feature maps from backbone/neck.

        Returns:
            candidates: per-image list of dicts with bbox, confidence, class_id.
            fcand: (B, N, 256) local features per candidate.
        """
        raise NotImplementedError(
            "Phase 2 (T2.3): implement multi-scale heads and RoI feature extraction."
        )

    def apply_soft_nms(
        self, boxes: torch.Tensor, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Gaussian Soft-NMS with configured sigma."""
        return soft_nms(boxes, scores, sigma=self.soft_nms_sigma)
