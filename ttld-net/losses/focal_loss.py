"""Focal loss for imbalanced Bosch dataset."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss with TTLD-Net defaults: gamma=1.5, alpha=0.75.

    Penalizes false negatives more than default gamma=2.0.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) unnormalized scores.
            targets: (N,) integer class labels.
        """
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1 - targets)
        loss = alpha_factor * (1 - pt) ** self.gamma * ce
        return loss.mean()
