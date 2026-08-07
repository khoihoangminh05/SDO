"""Gaussian Soft-NMS utilities."""

from __future__ import annotations

import torch


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float = 0.5,
    score_thresh: float = 0.05,
    max_keep: int = 300,
    iou_fn: str = "gaussian",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian Soft-NMS: score_new = score * exp(-IoU^2 / sigma).

    Args:
        boxes: (N, 4) in xyxy format.
        scores: (N,) confidence scores.
        sigma: Gaussian decay parameter.
        score_thresh: Minimum score to keep a box after decay.
        max_keep: Hard cap on number of retained boxes.
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4)), scores.new_zeros((0,))

    boxes = boxes.clone()
    scores = scores.clone()
    keep_boxes: list[torch.Tensor] = []
    keep_scores: list[torch.Tensor] = []

    while boxes.numel() > 0 and len(keep_boxes) < max_keep:
        top = int(torch.argmax(scores).item())
        top_score = float(scores[top].item())
        if top_score < score_thresh:
            break

        keep_boxes.append(boxes[top].unsqueeze(0))
        keep_scores.append(scores[top].unsqueeze(0))

        if boxes.shape[0] == 1:
            break

        remaining = torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device)
        remaining[top] = False
        others = boxes[remaining]
        other_scores = scores[remaining]
        ious = _box_iou_xyxy(boxes[top].unsqueeze(0), others).squeeze(0)

        if iou_fn == "gaussian":
            decay = torch.exp(-(ious**2) / max(sigma, 1e-6))
        else:
            decay = 1.0 - ious

        other_scores = other_scores * decay
        mask = other_scores >= score_thresh
        boxes = others[mask]
        scores = other_scores[mask]

    if not keep_boxes:
        return boxes.new_zeros((0, 4)), scores.new_zeros((0,))
    return torch.cat(keep_boxes, dim=0), torch.cat(keep_scores, dim=0)


def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for xyxy boxes."""
    if b.numel() == 0:
        return a.new_zeros((0,))
    lt = torch.max(a[:, None, :2], b[:, :2])
    rb = torch.min(a[:, None, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b - inter
    return inter / union.clamp(min=1e-6)
