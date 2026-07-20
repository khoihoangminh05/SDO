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
    Objectness uses positive cells + a capped random negative sample so
    720x1280 / stride-2 grids do not blow up memory or produce NaN grads.
    """

    def __init__(
        self,
        num_classes: int = 4,
        gamma: float = 1.5,
        alpha: float = 0.75,
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        box_weight: float = 1.0,
        max_obj_negatives: int = 4096,
        logit_clamp: float = 20.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.max_obj_negatives = max_obj_negatives
        self.logit_clamp = logit_clamp

    def _zero_loss(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Scalar zero that stays connected to the autograd graph."""
        return sum(tensor.sum() for tensor in tensors) * 0.0

    def forward(
        self,
        raw_outputs: list[dict[str, Any]],
        targets: list[torch.Tensor],
    ) -> torch.Tensor:
        if not raw_outputs:
            return torch.tensor(0.0)

        anchor = raw_outputs[0]["obj"]
        total = self._zero_loss(anchor)
        scales = 0

        for scale in raw_outputs:
            obj = scale["obj"]
            cls = scale["cls"]
            box = scale["box"]
            stride = int(scale["stride"])
            loss = self._scale_loss(obj, cls, box, stride, targets)
            if torch.isfinite(loss):
                total = total + loss
                scales += 1

        if scales == 0:
            parts = [scale["obj"] for scale in raw_outputs]
            parts += [scale["cls"] for scale in raw_outputs]
            parts += [scale["box"] for scale in raw_outputs]
            return self._zero_loss(*parts)
        return total

    def _sampled_obj_loss(
        self,
        obj: torch.Tensor,
        obj_target: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """BCE on positives + capped random negatives (avoids full-grid NaN)."""
        obj_flat = obj[:, 0].float().clamp(-self.logit_clamp, self.logit_clamp)
        target_flat = obj_target[:, 0].float()

        pos_idx = pos_mask.nonzero(as_tuple=False)
        neg_mask = ~pos_mask
        neg_idx = neg_mask.nonzero(as_tuple=False)

        if neg_idx.shape[0] > self.max_obj_negatives:
            pick = torch.randperm(neg_idx.shape[0], device=obj.device)[: self.max_obj_negatives]
            neg_idx = neg_idx[pick]

        if pos_idx.numel() == 0 and neg_idx.numel() == 0:
            return self._zero_loss(obj)

        if pos_idx.numel() == 0:
            sample_idx = neg_idx
        elif neg_idx.numel() == 0:
            sample_idx = pos_idx
        else:
            sample_idx = torch.cat([pos_idx, neg_idx], dim=0)

        b_idx = sample_idx[:, 0]
        y_idx = sample_idx[:, 1]
        x_idx = sample_idx[:, 2]
        logits = obj_flat[b_idx, y_idx, x_idx]
        labels = target_flat[b_idx, y_idx, x_idx]
        return F.binary_cross_entropy_with_logits(logits, labels)

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

        if not torch.isfinite(obj).all() or not torch.isfinite(cls).all() or not torch.isfinite(box).all():
            return self._zero_loss(obj, cls, box)

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
                if not all(torch.isfinite(torch.tensor([cx, cy, bw, bh]))):
                    continue

                grid_x = int(cx / stride)
                grid_y = int(cy / stride)
                if not (0 <= grid_x < width and 0 <= grid_y < height):
                    continue

                cls_id = int(class_id)
                if cls_id < 0 or cls_id >= self.num_classes:
                    cls_id = min(max(cls_id, 0), self.num_classes - 1)

                pos_mask[batch_idx, grid_y, grid_x] = True
                obj_target[batch_idx, 0, grid_y, grid_x] = 1.0
                cls_targets.append(cls_id)
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
                    torch.tensor([cx, cy, bw, bh], device=device, dtype=torch.float32)
                )

        obj_loss = self._sampled_obj_loss(obj, obj_target, pos_mask)

        if not cls_indices:
            return self.obj_weight * obj_loss

        cls_logits = torch.stack(
            [cls[b, :, y, x] for b, y, x in cls_indices],
            dim=0,
        ).float().clamp(-self.logit_clamp, self.logit_clamp)
        cls_labels = torch.tensor(cls_targets, device=device, dtype=torch.long)
        cls_loss = self._focal_ce(cls_logits, cls_labels)

        box_pred = torch.stack(box_preds, dim=0).float()
        box_tgt = torch.stack(box_targets, dim=0).float()
        box_loss = F.l1_loss(box_pred, box_tgt)

        total = (
            self.obj_weight * obj_loss
            + self.cls_weight * cls_loss
            + self.box_weight * box_loss
        )
        if not torch.isfinite(total):
            return self._zero_loss(obj, cls, box)
        return total

    def _focal_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce).clamp(min=1e-6, max=1.0 - 1e-6)
        alpha_factor = self.alpha * (targets > 0).float() + (1.0 - self.alpha) * (targets == 0).float()
        return (alpha_factor * (1 - pt) ** self.gamma * ce).mean()
