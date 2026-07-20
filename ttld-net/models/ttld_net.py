"""End-to-end TTLD-Net model assembly."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.backbones.yolo26 import YOLO26Backbone
from models.heads.implicit_topo import ImplicitTopologySampler
from models.heads.tiny_generator import TinyGenerator
from models.heads.verification import VerificationMLP
from models.necks.context_branch import SemanticContextBranch
from models.necks.fpn_panet import FPNPANet
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
            fcand_dim=256,
        )
        self.context_branch = SemanticContextBranch(c4=256, c5=256, c_cand=256)
        self.topo_sampler = ImplicitTopologySampler()
        self.verifier = VerificationMLP()
        self._stack_ready = False

    def _ensure_stack(self, device: torch.device, image_size: tuple[int, int] = (640, 640)) -> None:
        if self._stack_ready:
            return
        dims = self.backbone.channel_dims(image_size)
        self.neck = FPNPANet(
            in_channels=(dims["p1"], dims["p2"], dims["p3"], dims["p4"], dims["p5"]),
            out_channels=256,
        ).to(device)
        self.context_branch = SemanticContextBranch(
            c4=256,
            c5=256,
            c_cand=self.generator.fcand_dim,
            d_model=256,
        ).to(device)
        self._stack_ready = True

    def forward_stage1(self, images: torch.Tensor) -> dict[str, Any]:
        """Phase 2: backbone → neck → high-recall generator."""
        h, w = images.shape[-2], images.shape[-1]
        self._ensure_stack(images.device, (h, w))
        pyramids = self.backbone(images)
        p1, p2, p3, p4, p5 = self.neck(pyramids)
        candidates, fcand = self.generator(p1, p2, p3)
        return {
            "candidates": candidates,
            "fcand": fcand,
            "p4": p4,
            "p5": p5,
        }

    def forward_context(self, stage1: dict[str, Any]) -> dict[str, Any]:
        """Phase 3: project P4/P5 and fcand into unified 256-dim context space."""
        fctx_p4, fctx_p5, fcand_proj = self.context_branch(
            stage1["p4"], stage1["p5"], stage1["fcand"]
        )
        return {
            "fctx_p4": fctx_p4,
            "fctx_p5": fctx_p5,
            "fcand_proj": fcand_proj,
        }

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        """
        Forward pass.

        Phase 2–3 (shallow modes): stage-1 + context branch outputs.
        Phase 4+: adds topology sampler and verification when mode=full.
        """
        stage1 = self.forward_stage1(images)
        context = self.forward_context(stage1)

        outputs: dict[str, Any] = {
            **stage1,
            **context,
        }

        if self.cfg.model.mode in {"shallow", "shallow_focal", "baseline"}:
            return outputs

        from models.heads.implicit_topo import extract_reference_points

        batch_pq = []
        for image_candidates in stage1["candidates"]:
            batch_pq.append(extract_reference_points(image_candidates))
        pq = torch.stack(batch_pq, dim=0) if batch_pq else torch.zeros(0)
        fcand = context["fcand_proj"] if context["fcand_proj"] is not None else stage1["fcand"]
        zi = self.topo_sampler(fcand, [context["fctx_p4"], context["fctx_p5"]], pq)
        p_valid = self.verifier(fcand, zi)

        outputs["zi"] = zi
        outputs["p_valid"] = p_valid
        return outputs

    def compute_losses(
        self,
        outputs: dict[str, Any],
        targets: list[torch.Tensor],
        loss_fns: dict[str, nn.Module],
    ) -> dict[str, torch.Tensor]:
        """Combine L_det + lambda1*L_topology + lambda2*L_verify (Phase 6)."""
        raise NotImplementedError("Phase 6: implement multi-task loss combination.")
