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
        # Channel dims filled lazily after first backbone probe
        self.neck = FPNPANet()
        self.generator = TinyGenerator(
            conf_threshold=cfg.data.conf_threshold,
            soft_nms_sigma=cfg.data.soft_nms_sigma,
        )
        self.proj_p4 = FeatureProjection(256, 256)
        self.proj_p5 = FeatureProjection(256, 256)
        self.topo_sampler = ImplicitTopologySampler()
        self.verifier = VerificationMLP()
        self._neck_ready = False

    def _ensure_neck(self, device: torch.device) -> None:
        if self._neck_ready:
            return
        dims = self.backbone.channel_dims((640, 640))
        self.neck = FPNPANet(
            in_channels=(dims["p1"], dims["p2"], dims["p3"], dims["p4"], dims["p5"]),
            out_channels=256,
        ).to(device)
        self._neck_ready = True

    def forward_stage1(self, images: torch.Tensor) -> dict[str, Any]:
        """Phase 2 path: backbone → neck → high-recall generator."""
        self._ensure_neck(images.device)
        pyramids = self.backbone(images)
        p1, p2, p3, p4, p5 = self.neck(pyramids)
        candidates, fcand = self.generator(p1, p2, p3)
        return {
            "candidates": candidates,
            "fcand": fcand,
            "p4": p4,
            "p5": p5,
        }

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        """
        Full forward pass.

        Phase 2: stage-1 outputs. Phase 4+: topology + verification.
        """
        stage1 = self.forward_stage1(images)
        if self.cfg.model.mode in {"shallow", "shallow_focal", "baseline"}:
            return stage1

        fctx_p4 = self.proj_p4(stage1["p4"])
        fctx_p5 = self.proj_p5(stage1["p5"])

        from models.heads.implicit_topo import extract_reference_points

        batch_pq = []
        for image_candidates in stage1["candidates"]:
            batch_pq.append(extract_reference_points(image_candidates))
        pq = torch.stack(batch_pq, dim=0) if batch_pq else torch.zeros(0)
        zi = self.topo_sampler(stage1["fcand"], [fctx_p4, fctx_p5], pq)
        p_valid = self.verifier(stage1["fcand"], zi)

        return {
            "candidates": stage1["candidates"],
            "fcand": stage1["fcand"],
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
