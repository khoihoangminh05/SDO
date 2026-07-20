"""Semantic context branch — project P4/P5 (and optional fcand) to 256-dim."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.necks.fpn_panet import FeatureProjection


class SemanticContextBranch(nn.Module):
    """
    Module 2 — Semantic Context Branch (Phase 3).

    Projects deep pyramid features P4 (stride 16) and P5 (stride 32) into a
    shared 256-dim embedding space for the Implicit Topology Sampler (Phase 4).
    Optionally projects candidate features ``fcand`` to the same space.
    """

    def __init__(
        self,
        c4: int = 256,
        c5: int = 256,
        c_cand: int = 256,
        d_model: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj_p4 = FeatureProjection(c4, d_model)
        self.proj_p5 = FeatureProjection(c5, d_model)
        self.proj_cand = (
            nn.Linear(c_cand, d_model)
            if c_cand != d_model
            else nn.Identity()
        )

    def forward(
        self,
        p4: torch.Tensor,
        p5: torch.Tensor,
        fcand: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            p4: (B, C4, H4, W4) deep feature map, stride 16.
            p5: (B, C5, H5, W5) deep feature map, stride 32.
            fcand: optional (B, N, C_cand) local candidate features.

        Returns:
            fctx_p4: (B, 256, H4, W4)
            fctx_p5: (B, 256, H5, W5)
            fcand_proj: (B, N, 256) or None
        """
        fctx_p4 = self.proj_p4(p4)
        fctx_p5 = self.proj_p5(p5)
        fcand_proj = self.proj_cand(fcand) if fcand is not None else None
        return fctx_p4, fctx_p5, fcand_proj
