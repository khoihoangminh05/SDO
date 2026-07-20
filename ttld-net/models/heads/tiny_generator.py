"""High-recall candidate generator (Stage 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads.soft_nms import soft_nms


@dataclass
class Candidate:
    """Single traffic-light candidate."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    is_false_positive: bool = False


class _ScaleHead(nn.Module):
    """Per-scale prediction head: objectness, class, box, feature."""

    def __init__(self, in_channels: int, num_classes: int, fcand_dim: int) -> None:
        super().__init__()
        mid = max(in_channels, fcand_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.obj = nn.Conv2d(mid, 1, kernel_size=1)
        self.cls = nn.Conv2d(mid, num_classes, kernel_size=1)
        self.box = nn.Conv2d(mid, 4, kernel_size=1)
        self.feat = nn.Conv2d(mid, fcand_dim, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        return self.obj(h), self.cls(h), self.box(h), self.feat(h)


class TinyGenerator(nn.Module):
    """
    High-recall candidate generator on shallow features P1, P2, P3.

    Target: Recall > 95% with conf_threshold=0.05 and Soft-NMS.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (256, 256, 256),
        strides: tuple[int, int, int] = (2, 4, 8),
        num_classes: int = 4,
        conf_threshold: float = 0.05,
        soft_nms_sigma: float = 0.5,
        fcand_dim: int = 256,
        max_candidates: int = 2000,
    ) -> None:
        super().__init__()
        self.conf_threshold = conf_threshold
        self.soft_nms_sigma = soft_nms_sigma
        self.fcand_dim = fcand_dim
        self.num_classes = num_classes
        self.strides = strides
        self.max_candidates = max_candidates
        self.heads = nn.ModuleList(
            [_ScaleHead(ch, num_classes, fcand_dim) for ch in in_channels]
        )
        self.feature_proj = nn.Linear(fcand_dim, fcand_dim)

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        *,
        return_raw: bool = False,
    ) -> tuple[list[list[dict[str, Any]]], torch.Tensor] | tuple[
        list[list[dict[str, Any]]], torch.Tensor, list[dict[str, Any]]
    ]:
        """
        Args:
            p1, p2, p3: Shallow feature maps from backbone/neck.

        Returns:
            candidates: per-image list of dicts with bbox, confidence, class_id.
            fcand: (B, N, 256) local features per candidate (N = max over batch).
            raw_outputs: optional per-scale head logits for L_det training.
        """
        features = (p1, p2, p3)
        batch_size = p1.shape[0]
        device = p1.device

        per_image: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
        per_image_feats: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        raw_outputs: list[dict[str, Any]] = []

        for head, feat, stride in zip(self.heads, features, self.strides):
            obj, cls, box, emb = head(feat)
            if return_raw:
                raw_outputs.append(
                    {"obj": obj, "cls": cls, "box": box, "stride": stride}
                )
            self._decode_scale(
                obj, cls, box, emb, stride, per_image, per_image_feats
            )

        candidates: list[list[dict[str, Any]]] = []
        feat_lists: list[torch.Tensor] = []
        max_n = 0

        for b in range(batch_size):
            kept, kept_feat = self._nms_image(per_image[b], per_image_feats[b], device)
            candidates.append(kept)
            feat_lists.append(kept_feat)
            max_n = max(max_n, kept_feat.shape[0])

        max_n = max(max_n, 1)
        fcand = torch.zeros(batch_size, max_n, self.fcand_dim, device=device)
        for b, feats in enumerate(feat_lists):
            if feats.numel() == 0:
                continue
            n = min(feats.shape[0], max_n)
            fcand[b, :n] = self.feature_proj(feats[:n])

        if return_raw:
            return candidates, fcand, raw_outputs
        return candidates, fcand

    def _decode_scale(
        self,
        obj: torch.Tensor,
        cls: torch.Tensor,
        box: torch.Tensor,
        emb: torch.Tensor,
        stride: int,
        per_image: list[list[dict[str, Any]]],
        per_image_feats: list[list[torch.Tensor]],
    ) -> None:
        batch, _, height, width = obj.shape
        obj_p = obj.sigmoid()
        cls_p = cls.sigmoid()
        scores, class_ids = cls_p.max(dim=1)
        conf = obj_p.squeeze(1) * scores

        # box: tx,ty,tw,th → xyxy in pixel space
        gy, gx = torch.meshgrid(
            torch.arange(height, device=obj.device),
            torch.arange(width, device=obj.device),
            indexing="ij",
        )
        tx, ty, tw, th = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        cx = (gx[None] + tx.sigmoid()) * stride
        cy = (gy[None] + ty.sigmoid()) * stride
        bw = tw.exp().clamp(max=50.0) * stride
        bh = th.exp().clamp(max=50.0) * stride
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        for b in range(batch):
            mask = conf[b] >= self.conf_threshold
            if not mask.any():
                continue
            ys, xs = torch.where(mask)
            # Cap per-scale density for memory
            if ys.numel() > self.max_candidates:
                topk = torch.topk(conf[b][mask], self.max_candidates).indices
                ys, xs = ys[topk], xs[topk]

            for y, x in zip(ys.tolist(), xs.tolist()):
                cx_v = float(cx[b, y, x].detach())
                cy_v = float(cy[b, y, x].detach())
                bw_v = float(bw[b, y, x].detach())
                bh_v = float(bh[b, y, x].detach())
                per_image[b].append(
                    {
                        "bbox": (cx_v, cy_v, bw_v, bh_v),  # xywh (pixel)
                        "bbox_xyxy": (
                            float(x1[b, y, x].detach()),
                            float(y1[b, y, x].detach()),
                            float(x2[b, y, x].detach()),
                            float(y2[b, y, x].detach()),
                        ),
                        "confidence": float(conf[b, y, x].detach()),
                        "class_id": int(class_ids[b, y, x].detach()),
                    }
                )
                per_image_feats[b].append(emb[b, :, y, x])

    def _nms_image(
        self,
        cands: list[dict[str, Any]],
        feats: list[torch.Tensor],
        device: torch.device,
    ) -> tuple[list[dict[str, Any]], torch.Tensor]:
        if not cands:
            return [], torch.zeros(0, self.fcand_dim, device=device)

        boxes = torch.tensor(
            [c["bbox_xyxy"] for c in cands], device=device, dtype=torch.float32
        )
        scores = torch.tensor([c["confidence"] for c in cands], device=device, dtype=torch.float32)
        feat_mat = torch.stack(feats, dim=0)

        keep_boxes, keep_scores = soft_nms(boxes, scores, sigma=self.soft_nms_sigma)
        if keep_boxes.numel() == 0:
            return [], torch.zeros(0, self.fcand_dim, device=device)

        kept: list[dict[str, Any]] = []
        kept_feats: list[torch.Tensor] = []
        used = set()
        for kb, ks in zip(keep_boxes, keep_scores):
            ious = _pairwise_iou(kb.unsqueeze(0), boxes).squeeze(0)
            order = torch.argsort(ious, descending=True)
            for idx in order.tolist():
                if idx in used:
                    continue
                used.add(idx)
                x1, y1, x2, y2 = (float(v) for v in kb.tolist())
                kept.append(
                    {
                        "bbox": ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
                        "confidence": float(ks.item()),
                        "class_id": cands[idx]["class_id"],
                    }
                )
                kept_feats.append(feat_mat[idx])
                break

        return kept, torch.stack(kept_feats, dim=0) if kept_feats else torch.zeros(
            0, self.fcand_dim, device=device
        )

    def apply_soft_nms(
        self, boxes: torch.Tensor, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Gaussian Soft-NMS with configured sigma."""
        return soft_nms(boxes, scores, sigma=self.soft_nms_sigma)


def _pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lt = torch.max(a[:, None, :2], b[:, :2])
    rb = torch.min(a[:, None, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b - inter).clamp(min=1e-6)
