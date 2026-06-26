"""End-to-end TTLD-Net model assembly."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.backbones.yolo26 import YOLO26Backbone
from models.heads.implicit_topo import ImplicitTopologySampler
from models.heads.tiny_generator import TinyGenerator
from models.heads.verification import VerificationMLP
from models.necks.fpn_panet import FPNPANet, FeatureProjection
from utils.config import TTLDConfig


class TTLDNet(nn.Module):
    """Topology-Aware Tiny Traffic Light Detection Network."""

    def __init__(self, cfg: TTLDConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = YOLO26Backbone()
        self.neck = FPNPANet()
        self.generator = TinyGenerator(
            conf_threshold=cfg.data.conf_threshold,
            soft_nms_sigma=cfg.data.soft_nms_sigma,
        )
        self.proj_p4 = FeatureProjection(512, 256)
        self.proj_p5 = FeatureProjection(1024, 256)
        self.topo_sampler = ImplicitTopologySampler()
        self.verifier = VerificationMLP()

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        """
        Full forward pass.

        Returns dict with candidates, fcand, zi, p_valid when all phases implemented.
        """
        p1, p2, p3, p4, p5 = self.backbone(images)
        p1, p2, p3, p4, p5 = self.neck((p1, p2, p3, p4, p5))
        candidates, fcand = self.generator(p1, p2, p3)

        fctx_p4 = self.proj_p4(p4)
        fctx_p5 = self.proj_p5(p5)

        from models.heads.implicit_topo import extract_reference_points

        batch_pq = []
        for image_candidates in candidates:
            batch_pq.append(extract_reference_points(image_candidates))
        # Padding/stacking handled in Phase 4
        pq = torch.stack(batch_pq, dim=0) if batch_pq else torch.zeros(0)
        zi = self.topo_sampler(fcand, [fctx_p4, fctx_p5], pq)
        p_valid = self.verifier(fcand, zi)

        return {
            "candidates": candidates,
            "fcand": fcand,
            "zi": zi,
            "p_valid": p_valid,
        }

    def compute_losses(
        self,
        outputs: dict[str, Any],
        targets: list[torch.Tensor],
        loss_fns: dict[str, nn.Module],
    ) -> dict[str, torch.Tensor]:
        """Combine L_det + lambda1*L_topology + lambda2*L_verify (Phase 6)."""
        raise NotImplementedError("Phase 6: implement multi-task loss combination.")
