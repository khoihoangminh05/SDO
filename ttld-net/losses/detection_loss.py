"""Detection loss (L_det) for TinyGenerator multi-scale heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """
    YOLO-style assignment on P1/P2/P3 grid cells.

    Positive cell = GT center falls inside the cell at each stride.
    Uses focal-modulated classification and L1 box regression on positives.
    """

    def __init__(
        self,
        num_classes: int = 4,
        gamma: float = 1.5,
        alpha: float = 0.75,
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        box_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.box_weight = box_weight

    def forward(
        self,
        raw_outputs: list[dict[str, Any]],
        targets: list[torch.Tensor],
    ) -> torch.Tensor:
        if not raw_outputs:
            return torch.tensor(0.0)

        device = raw_outputs[0]["obj"].device
        total = torch.tensor(0.0, device=device)
        scales = 0

        for scale in raw_outputs:
            obj = scale["obj"]
            cls = scale["cls"]
            box = scale["box"]
            stride = int(scale["stride"])
            loss = self._scale_loss(obj, cls, box, stride, targets)
            total = total + loss
            scales += 1

        return total / max(scales, 1)

    def _scale_loss(
        self,
        obj: torch.Tensor,
        cls: torch.Tensor,
        box: torch.Tensor,
        stride: int,
        targets: list[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, _, height, width = obj.shape
        device = obj.device

        obj_target = torch.zeros_like(obj)
        pos_mask = torch.zeros(batch_size, height, width, dtype=torch.bool, device=device)
        cls_targets: list[int] = []
        cls_indices: list[tuple[int, int, int]] = []
        box_preds: list[torch.Tensor] = []
        box_targets: list[torch.Tensor] = []

        gy, gx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )

        for batch_idx, gt in enumerate(targets):
            if gt.numel() == 0:
                continue
            for box_gt in gt:
                cx, cy, bw, bh, class_id = box_gt.tolist()
                grid_x = int(cx / stride)
                grid_y = int(cy / stride)
                if not (0 <= grid_x < width and 0 <= grid_y < height):
                    continue

                pos_mask[batch_idx, grid_y, grid_x] = True
                obj_target[batch_idx, 0, grid_y, grid_x] = 1.0
                cls_targets.append(int(class_id))
                cls_indices.append((batch_idx, grid_y, grid_x))

                tx, ty, tw, th = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
                tw_safe = tw[batch_idx, grid_y, grid_x].float().clamp(-4.0, 4.0)
                th_safe = th[batch_idx, grid_y, grid_x].float().clamp(-4.0, 4.0)
                pred_cx = (gx[grid_y, grid_x] + tx[batch_idx, grid_y, grid_x].float().sigmoid()) * stride
                pred_cy = (gy[grid_y, grid_x] + ty[batch_idx, grid_y, grid_x].float().sigmoid()) * stride
                pred_bw = tw_safe.exp().clamp(max=50.0) * stride
                pred_bh = th_safe.exp().clamp(max=50.0) * stride
                box_preds.append(torch.stack([pred_cx, pred_cy, pred_bw, pred_bh]))
                box_targets.append(
                    torch.tensor([cx, cy, bw, bh], device=device, dtype=box.dtype)
                )

        obj_loss = F.binary_cross_entropy_with_logits(obj.float(), obj_target.float())

        if not cls_indices:
            return self.obj_weight * obj_loss

        cls_logits = torch.stack(
            [cls[b, :, y, x] for b, y, x in cls_indices],
            dim=0,
        ).float()
        cls_labels = torch.tensor(cls_targets, device=device, dtype=torch.long)
        cls_loss = self._focal_ce(cls_logits, cls_labels)

        box_pred = torch.stack(box_preds, dim=0)
        box_tgt = torch.stack(box_targets, dim=0)
        box_loss = F.l1_loss(box_pred, box_tgt)

        return (
            self.obj_weight * obj_loss
            + self.cls_weight * cls_loss
            + self.box_weight * box_loss
        )

    def _focal_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_factor = self.alpha * (targets > 0).float() + (1.0 - self.alpha) * (targets == 0).float()
        return (alpha_factor * (1 - pt) ** self.gamma * ce).mean()
