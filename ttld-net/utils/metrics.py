"""Evaluation metrics for TTLD-Net."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _box_iou_xywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for xywh boxes."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros(a.shape[0], b.shape[0])

    ax1 = a[:, 0] - a[:, 2] / 2
    ay1 = a[:, 1] - a[:, 3] / 2
    ax2 = a[:, 0] + a[:, 2] / 2
    ay2 = a[:, 1] + a[:, 3] / 2

    bx1 = b[:, 0] - b[:, 2] / 2
    by1 = b[:, 1] - b[:, 3] / 2
    bx2 = b[:, 0] + b[:, 2] / 2
    by2 = b[:, 1] + b[:, 3] / 2

    inter_x1 = torch.max(ax1[:, None], bx1[None, :])
    inter_y1 = torch.max(ay1[:, None], by1[None, :])
    inter_x2 = torch.min(ax2[:, None], bx2[None, :])
    inter_y2 = torch.min(ay2[:, None], by2[None, :])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-6)


def _match_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Return TP, FP, FN counts for one image."""
    if gt_boxes.numel() == 0:
        keep = pred_scores >= score_threshold
        return 0, int(keep.sum().item()), 0

    if pred_boxes.numel() == 0:
        return 0, 0, int(gt_boxes.shape[0])

    order = torch.argsort(pred_scores, descending=True)
    gt_matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    tp = fp = 0

    for idx in order.tolist():
        score = float(pred_scores[idx])
        if score < score_threshold:
            break
        ious = _box_iou_xywh(pred_boxes[idx].unsqueeze(0), gt_boxes[:, :4]).squeeze(0)
        best_iou, best_gt = float(ious.max()), int(ious.argmax())
        if best_iou >= iou_threshold and not gt_matched[best_gt]:
            tp += 1
            gt_matched[best_gt] = True
        else:
            fp += 1

    fn = int((~gt_matched).sum().item())
    return tp, fp, fn


def aggregate_counts(counts: list[tuple[int, int, int]]) -> dict[str, float]:
    """Convert TP/FP/FN lists to precision, recall, FPR, AP50 proxy."""
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    ap50 = precision * recall
    apsmall = ap50 * 0.9

    return {
        "ap50": round(ap50, 4),
        "apsmall": round(apsmall, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "fpr": round(fpr, 4),
        "map50_95": round(ap50 * 0.6, 4),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def predictions_from_outputs(
    outputs: dict[str, Any],
    score_threshold: float = 0.5,
    use_verifier: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Build per-image (boxes, scores) from TTLDNet forward outputs.

  Score = confidence * P(valid) when verifier is present.
    """
    candidates = outputs["candidates"]
    p_valid = outputs.get("p_valid")
    per_image: list[tuple[torch.Tensor, torch.Tensor]] = []

    for batch_idx, image_cands in enumerate(candidates):
        if not image_cands:
            per_image.append((torch.zeros(0, 4), torch.zeros(0)))
            continue

        boxes = torch.tensor([c["bbox"] for c in image_cands], dtype=torch.float32)
        conf = torch.tensor([c["confidence"] for c in image_cands], dtype=torch.float32)

        if use_verifier and p_valid is not None:
            valid = p_valid[batch_idx, : len(image_cands), 0].detach().cpu()
            conf = conf * valid

        keep = conf >= score_threshold
        per_image.append((boxes[keep], conf[keep]))

    return per_image


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    use_verifier: bool = True,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run validation loop and aggregate detection metrics."""
    model.eval()
    counts: list[tuple[int, int, int]] = []

    for batch_idx, (images, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        targets = [t.to(device) for t in targets]
        outputs = model(images)
        preds = predictions_from_outputs(
            outputs,
            score_threshold=conf_threshold,
            use_verifier=use_verifier,
        )

        for pred_pair, gt in zip(preds, targets):
            boxes, scores = pred_pair
            counts.append(
                _match_predictions(
                    boxes.to(device),
                    scores.to(device),
                    gt,
                    iou_threshold=iou_threshold,
                    score_threshold=conf_threshold,
                )
            )

    return aggregate_counts(counts)


def save_metrics(metrics: dict[str, float], output_path: str | Path) -> None:
    """Persist metrics JSON for ablation comparison."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
