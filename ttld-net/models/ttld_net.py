"""End-to-end TTLD-Net model assembly."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.backbones.yolo26 import YOLO26Backbone
from models.heads.implicit_topo import (
    ImplicitTopologySampler,
    build_reference_points_batch,
)
from models.heads.tiny_generator import TinyGenerator
from losses.infonce_loss import InfoNCELoss
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
            max_candidates=cfg.training.max_candidates,
        )
        self.context_branch = SemanticContextBranch(c4=256, c5=256, c_cand=256)
        topo_cfg = cfg.raw.get("topology", {}) if cfg.raw else {}
        self.topo_sampler = ImplicitTopologySampler(
            d_model=int(topo_cfg.get("d_model", 256)),
            n_heads=int(topo_cfg.get("n_heads", 8)),
            n_points=int(topo_cfg.get("n_points", 4)),
            num_levels=2,
        )
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
        gen_out = self.generator(p1, p2, p3, return_raw=True)
        candidates, fcand, raw_outputs = gen_out
        return {
            "candidates": candidates,
            "fcand": fcand,
            "raw_outputs": raw_outputs,
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

        fcand = context["fcand_proj"] if context["fcand_proj"] is not None else stage1["fcand"]
        image_size = self.cfg.data.image_size
        pq = build_reference_points_batch(
            stage1["candidates"],
            image_size=image_size,
            max_n=fcand.shape[1],
            device=fcand.device,
            dtype=fcand.dtype,
        )
        zi = self.topo_sampler(fcand, [context["fctx_p4"], context["fctx_p5"]], pq)
        outputs["zi"] = zi

        if self.cfg.model.mode == "full":
            valid_logits = self.verifier(fcand, zi)
            outputs["valid_logits"] = valid_logits
            outputs["p_valid"] = torch.sigmoid(valid_logits)

        return outputs

    def compute_losses(
        self,
        outputs: dict[str, Any],
        targets: list[torch.Tensor],
        loss_fns: dict[str, nn.Module] | None = None,
        lambda1: float | None = None,
        lambda2: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Phase 6: L_det + λ1·L_topology + λ2·L_verify.

        Uses ``lambda1`` / ``lambda2`` from config to weight contrastive and
        verification losses.
        """
        from losses.detection_loss import DetectionLoss
        from models.heads.verification import (
            compute_topology_loss,
            label_candidates_batch,
            verification_bce_loss,
        )

        loss_fns = loss_fns or {}
        losses: dict[str, torch.Tensor] = {}
        total = outputs["fcand"].sum() * 0.0
        lambda1 = self.cfg.loss.lambda1 if lambda1 is None else lambda1
        lambda2 = self.cfg.loss.lambda2 if lambda2 is None else lambda2

        det_fn = loss_fns.get("detection")
        if det_fn is None:
            det_fn = DetectionLoss(
                gamma=self.cfg.loss.focal_gamma,
                alpha=self.cfg.loss.focal_alpha,
            )
        if outputs.get("raw_outputs"):
            loss_det = det_fn(outputs["raw_outputs"], targets)
            losses["det"] = loss_det
            total = total + loss_det

        max_n = outputs["fcand"].shape[1]
        labels, confidences = label_candidates_batch(
            outputs["candidates"],
            targets,
            max_n=max_n,
            iou_threshold=0.5,
        )
        device = outputs["fcand"].device
        labels = labels.to(device)
        confidences = confidences.to(device)

        if "valid_logits" in outputs and lambda2 > 0:
            loss_verify = verification_bce_loss(outputs["valid_logits"], labels)
            losses["verify"] = loss_verify
            total = total + lambda2 * loss_verify

        if "zi" in outputs and lambda1 > 0:
            infonce = loss_fns.get("infonce")
            if infonce is None:
                infonce = InfoNCELoss(temperature=self.cfg.loss.infonce_temperature).to(device)
            loss_topo = compute_topology_loss(
                outputs["zi"],
                labels,
                confidences,
                infonce,
                n_hard=self.cfg.loss.n_hard_negatives,
            )
            losses["topology"] = loss_topo
            total = total + lambda1 * loss_topo

        losses["total"] = total
        return losses
