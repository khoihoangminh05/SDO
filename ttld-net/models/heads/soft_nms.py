"""Gaussian Soft-NMS utilities."""

from __future__ import annotations

import torch


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float = 0.5,
    score_thresh: float = 0.001,
    iou_fn: str = "gaussian",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian Soft-NMS: score_new = score * exp(-IoU^2 / sigma).

    Args:
        boxes: (N, 4) in xyxy format.
        scores: (N,) confidence scores.
        sigma: Gaussian decay parameter.
        score_thresh: Minimum score to keep a box.
    """
    if boxes.numel() == 0:
        return boxes, scores

    boxes = boxes.clone()
    scores = scores.clone()
    keep_boxes: list[torch.Tensor] = []
    keep_scores: list[torch.Tensor] = []

    while boxes.numel() > 0:
        top = int(torch.argmax(scores).item())
        keep_boxes.append(boxes[top].unsqueeze(0))
        keep_scores.append(scores[top].unsqueeze(0))

        if boxes.shape[0] == 1:
            break

        remaining = [i for i in range(boxes.shape[0]) if i != top]
        ref = boxes[top]
        others = boxes[remaining]
        ious = _box_iou_xyxy(ref.unsqueeze(0), others).squeeze(0)

        if iou_fn == "gaussian":
            decay = torch.exp(-(ious**2) / sigma)
        else:
            decay = 1.0 - ious

        scores[remaining] *= decay
        mask = scores[remaining] >= score_thresh
        boxes = torch.cat([boxes[top].unsqueeze(0), others[mask]], dim=0)
        scores = torch.cat([scores[top].unsqueeze(0), scores[remaining][mask]], dim=0)
        boxes = boxes[1:]
        scores = scores[1:]

    return torch.cat(keep_boxes, dim=0), torch.cat(keep_scores, dim=0)


def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for xyxy boxes."""
    lt = torch.max(a[:, None, :2], b[:, :2])
    rb = torch.min(a[:, None, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b - inter
    return inter / union.clamp(min=1e-6)
