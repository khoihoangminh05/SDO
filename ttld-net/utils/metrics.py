"""Evaluation metrics for TTLD-Net (VOC-style AP, size bins, mAP@0.5:0.95)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

# BSTLD-oriented size bins on max(w, h) in pixels (after resize).
_SIZE_BINS: dict[str, tuple[float, float]] = {
    "tiny": (0.0, 8.0),
    "small": (8.0, 16.0),
    "medium": (16.0, 32.0),
}


def _box_iou_xywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for xywh boxes."""
    if a.numel() == 0 or b.numel() == 0:
        device = a.device if a.numel() else b.device
        return torch.zeros(a.shape[0], b.shape[0], device=device)

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
    score_threshold: float = 0.0,
) -> tuple[list[tuple[float, int]], int]:
    """
    Greedy score-order matching for one image.

    Returns:
        matches: list of (score, is_tp)
        num_gt: number of ground-truth boxes used
    """
    num_gt = int(gt_boxes.shape[0]) if gt_boxes.numel() else 0
    if pred_boxes.numel() == 0:
        return [], num_gt

    order = torch.argsort(pred_scores, descending=True)
    gt_matched = torch.zeros(num_gt, dtype=torch.bool, device=pred_boxes.device)
    matches: list[tuple[float, int]] = []

    for idx in order.tolist():
        score = float(pred_scores[idx])
        if score < score_threshold:
            break
        if num_gt == 0:
            matches.append((score, 0))
            continue
        ious = _box_iou_xywh(pred_boxes[idx].unsqueeze(0), gt_boxes[:, :4]).squeeze(0)
        best_iou, best_gt = float(ious.max()), int(ious.argmax())
        if best_iou >= iou_threshold and not bool(gt_matched[best_gt]):
            matches.append((score, 1))
            gt_matched[best_gt] = True
        else:
            matches.append((score, 0))

    return matches, num_gt


def average_precision(scores_and_tp: list[tuple[float, int]], num_gt: int) -> float:
    """VOC-style continuous AP from ranked (score, is_tp) detections."""
    if num_gt <= 0 or not scores_and_tp:
        return 0.0

    ranked = sorted(scores_and_tp, key=lambda x: x[0], reverse=True)
    tp_cum = fp_cum = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for _, is_tp in ranked:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        recalls.append(tp_cum / num_gt)
        precisions.append(tp_cum / (tp_cum + fp_cum))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    prev_recall = 0.0
    for recall, precision in zip(recalls, precisions):
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return float(ap)


def _gt_size_mask(gt: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Select GT rows whose max(w, h) is in [lo, hi)."""
    if gt.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=gt.device)
    side = torch.maximum(gt[:, 2], gt[:, 3])
    return (side >= lo) & (side < hi)


def _ap_for_images(
    images: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    iou_threshold: float,
    gt_mask_fn=None,
) -> float:
    all_matches: list[tuple[float, int]] = []
    total_gt = 0
    for boxes, scores, gt in images:
        if gt_mask_fn is not None:
            mask = gt_mask_fn(gt)
            gt = gt[mask]
        matches, num_gt = _match_predictions(
            boxes, scores, gt, iou_threshold=iou_threshold, score_threshold=0.0
        )
        all_matches.extend(matches)
        total_gt += num_gt
    return average_precision(all_matches, total_gt)


def aggregate_matches(
    all_matches: list[tuple[float, int]],
    total_gt: int,
    operating_conf: float = 0.05,
) -> dict[str, float]:
    """Operating-point stats + true AP50 from global matches (legacy helper)."""
    ap50 = average_precision(all_matches, total_gt)

    tp = fp = 0
    for score, is_tp in all_matches:
        if score < operating_conf:
            continue
        if is_tp:
            tp += 1
        else:
            fp += 1
    fn = max(total_gt - tp, 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    return {
        "ap50": round(ap50, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "fpr": round(fpr, 4),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "num_gt": float(total_gt),
        "num_dets": float(len(all_matches)),
    }


def aggregate_counts(counts: list[tuple[int, int, int]]) -> dict[str, float]:
    """Single-threshold count aggregator (precision / recall / F1-proxy)."""
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "ap50": round(f1, 4),  # F1 proxy when only counts are available
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "fpr": round(fpr, 4),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def predictions_from_outputs(
    outputs: dict[str, Any],
    score_threshold: float = 0.5,
    use_verifier: bool = True,
    max_dets: int = 100,
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
        boxes = boxes[keep]
        conf = conf[keep]
        if conf.numel() > max_dets:
            topk = torch.topk(conf, max_dets).indices
            boxes = boxes[topk]
            conf = conf[topk]
        per_image.append((boxes, conf))

    return per_image


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    conf_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    use_verifier: bool = True,
    max_batches: int | None = None,
    max_dets: int = 100,
) -> dict[str, float]:
    """
    Validation metrics:

    - ``ap50``: VOC-style AP @ IoU=0.5
    - ``map50_95``: mean AP over IoU in {0.50, 0.55, …, 0.95}
    - ``ap_tiny`` / ``ap_small`` / ``ap_medium``: AP50 on BSTLD size bins
    - ``precision`` / ``recall``: single operating point at ``conf_threshold``
    """
    model.eval()
    images_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    all_matches_50: list[tuple[float, int]] = []
    total_gt = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        targets = [t.to(device) for t in targets]
        outputs = model(images)
        preds = predictions_from_outputs(
            outputs,
            score_threshold=min(conf_threshold, 0.01),
            use_verifier=use_verifier,
            max_dets=max_dets,
        )

        for pred_pair, gt in zip(preds, targets):
            boxes, scores = pred_pair
            boxes = boxes.to(device)
            scores = scores.to(device)
            images_cache.append((boxes, scores, gt))
            matches, num_gt = _match_predictions(
                boxes, scores, gt, iou_threshold=iou_threshold, score_threshold=0.0
            )
            all_matches_50.extend(matches)
            total_gt += num_gt

    metrics = aggregate_matches(all_matches_50, total_gt, operating_conf=conf_threshold)

    # True mAP@0.5:0.95 (COCO-style mean over IoU thresholds)
    iou_thrs = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    aps = [_ap_for_images(images_cache, thr) for thr in iou_thrs]
    metrics["map50_95"] = round(sum(aps) / len(aps), 4)

    # Size-binned AP50 (honest APsmall ≈ tiny∪small for BSTLD)
    for name, (lo, hi) in _SIZE_BINS.items():
        metrics[f"ap_{name}"] = round(
            _ap_for_images(
                images_cache,
                iou_threshold=0.5,
                gt_mask_fn=lambda g, lo=lo, hi=hi: _gt_size_mask(g, lo, hi),
            ),
            4,
        )
    metrics["apsmall"] = round(
        _ap_for_images(
            images_cache,
            iou_threshold=0.5,
            gt_mask_fn=lambda g: _gt_size_mask(g, 0.0, 16.0),
        ),
        4,
    )
    metrics["eval_iou"] = float(iou_threshold)
    metrics["operating_conf"] = float(conf_threshold)
    return metrics


def save_metrics(metrics: dict[str, float], output_path: str | Path) -> None:
    """Persist metrics JSON for ablation comparison."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
