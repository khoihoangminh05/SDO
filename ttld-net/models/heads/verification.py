"""Verification head and hard-negative mining."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class VerificationMLP(nn.Module):
    """Fuse local and topology features to predict P(Valid)."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, fcand: torch.Tensor, zi: torch.Tensor) -> torch.Tensor:
        vi = torch.cat([fcand, zi], dim=-1)
        return self.mlp(vi)


def mine_hard_negatives(candidates: list[dict[str, Any]], n_hard: int = 32) -> list[dict[str, Any]]:
    """
    Select false-positive candidates with highest confidence as hard negatives.

    Requires `is_false_positive` flag on each candidate dict.
    """
    fp_candidates = [c for c in candidates if c.get("is_false_positive")]
    fp_candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return fp_candidates[:n_hard]
