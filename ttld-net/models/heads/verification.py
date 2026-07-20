"""Verification head, hard-negative mining, and Phase 5 loss helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.infonce_loss import InfoNCELoss


class VerificationMLP(nn.Module):
    """Fuse local and topology features to predict P(Valid)."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, fcand: torch.Tensor, zi: torch.Tensor) -> torch.Tensor:
        vi = torch.cat([fcand, zi], dim=-1)
        return self.mlp(vi)


def _box_iou_xywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for xywh boxes. a: (N, 4), b: (M, 4) -> (N, M)."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros(a.shape[0], b.shape[0], device=a.device, dtype=a.dtype)

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
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def match_candidates_to_gt(
    candidates: list[dict[str, Any]],
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Label each candidate as TP (1) or FP (0) via max IoU against ground-truth boxes.

    Args:
        candidates: list of dicts with ``bbox`` as (x_center, y_center, w, h) in pixels.
        gt_boxes: (G, 5) tensor — x_center, y_center, w, h, class_id.
    """
    if not candidates:
        return torch.zeros(0, dtype=torch.float32)

    cand_boxes = torch.tensor(
        [c["bbox"] for c in candidates],
        dtype=torch.float32,
        device=gt_boxes.device if gt_boxes.numel() else "cpu",
    )
    if gt_boxes.numel() == 0:
        return torch.zeros(len(candidates), dtype=torch.float32, device=cand_boxes.device)

    gt_xywh = gt_boxes[:, :4].to(cand_boxes.device)
    max_iou = _box_iou_xywh(cand_boxes, gt_xywh).max(dim=1).values
    return (max_iou >= iou_threshold).float()


def label_candidates_batch(
    candidates: list[list[dict[str, Any]]],
    targets: list[torch.Tensor],
    max_n: int,
    iou_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build padded validity labels and confidence scores aligned with fcand.

    Returns:
        labels: (B, max_n) — 1=TP, 0=FP, -1=padding (ignored in loss).
        confidences: (B, max_n)
    """
    batch_size = len(candidates)
    labels = torch.full((batch_size, max_n), -1.0, dtype=torch.float32)
    confidences = torch.zeros(batch_size, max_n, dtype=torch.float32)

    for batch_idx, (image_cands, gt) in enumerate(zip(candidates, targets)):
        n = min(len(image_cands), max_n)
        for cand_idx in range(n):
            confidences[batch_idx, cand_idx] = float(image_cands[cand_idx]["confidence"])

        if n == 0:
            continue

        cand_labels = match_candidates_to_gt(image_cands[:n], gt, iou_threshold)
        labels[batch_idx, :n] = cand_labels.to(labels.device)

        for cand_idx in range(n):
            image_cands[cand_idx]["is_false_positive"] = cand_labels[cand_idx].item() < 0.5

    return labels, confidences


def mine_hard_negatives(
    candidates: list[dict[str, Any]],
    n_hard: int = 32,
) -> list[dict[str, Any]]:
    """
    Select false-positive candidates with highest confidence as hard negatives.

    Requires ``is_false_positive`` on each candidate (set by ``label_candidates_batch``).
    """
    fp_candidates = [c for c in candidates if c.get("is_false_positive")]
    fp_candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return fp_candidates[:n_hard]


def build_positive_embeddings(zi: torch.Tensor, tp_mask: torch.Tensor) -> torch.Tensor:
    """
    Build positive topology embeddings for InfoNCE.

    For each true positive, ``z_pos`` is the embedding of another TP in the same
    image (cyclic shift). Single-TP images use a lightly perturbed self copy.
    """
    batch_size, _, embed_dim = zi.shape
    z_pos = zi.clone()

    for batch_idx in range(batch_size):
        tp_indices = tp_mask[batch_idx].nonzero(as_tuple=True)[0]
        if tp_indices.numel() == 0:
            continue
        if tp_indices.numel() == 1:
            idx = int(tp_indices[0])
            z_pos[batch_idx, idx] = zi[batch_idx, idx] + 0.01 * torch.randn(
                embed_dim, device=zi.device, dtype=zi.dtype
            )
            continue
        rolled = zi[batch_idx, tp_indices].roll(1, dims=0)
        z_pos[batch_idx, tp_indices] = rolled

    return z_pos


def _hard_negatives_for_image(
    zi: torch.Tensor,
    fp_mask_row: torch.Tensor,
    confidences_row: torch.Tensor,
    n_hard: int,
) -> torch.Tensor:
    """Return (M, D) hard-negative embeddings for one image."""
    embed_dim = zi.shape[-1]
    fp_indices = fp_mask_row.nonzero(as_tuple=True)[0]

    if fp_indices.numel() == 0:
        return torch.randn(n_hard, embed_dim, device=zi.device, dtype=zi.dtype)

    order = torch.argsort(confidences_row[fp_indices], descending=True)
    fp_indices = fp_indices[order]
    fp_emb = zi[fp_indices]

    if fp_emb.shape[0] < n_hard:
        reps = (n_hard + fp_emb.shape[0] - 1) // fp_emb.shape[0]
        fp_emb = fp_emb.repeat(reps, 1)[:n_hard]
    else:
        fp_emb = fp_emb[:n_hard]

    return fp_emb


def prepare_infonce_tensors(
    zi: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    n_hard: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Gather (K, D), (K, D), (K, M, D) tensors for true-positive anchors only.

    Returns None when no true positives exist in the batch.
    """
    tp_mask = labels == 1
    fp_mask = labels == 0

    if tp_mask.sum() == 0:
        return None

    z_pos_full = build_positive_embeddings(zi, tp_mask)
    anchors: list[torch.Tensor] = []
    positives: list[torch.Tensor] = []
    negatives: list[torch.Tensor] = []

    batch_size = zi.shape[0]
    for batch_idx in range(batch_size):
        tp_indices = tp_mask[batch_idx].nonzero(as_tuple=True)[0]
        if tp_indices.numel() == 0:
            continue

        fp_neg = _hard_negatives_for_image(
            zi[batch_idx],
            fp_mask[batch_idx],
            confidences[batch_idx],
            n_hard,
        )

        for idx in tp_indices:
            anchors.append(zi[batch_idx, idx])
            positives.append(z_pos_full[batch_idx, idx])
            negatives.append(fp_neg)

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def verification_bce_loss(valid_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """BCE over valid (non-padding) candidate slots; logits are AMP-safe."""
    preds = valid_logits.squeeze(-1)
    if labels.shape != preds.shape:
        if labels.dim() == 1 and preds.dim() == 2 and labels.shape[0] == preds.shape[1]:
            labels = labels.unsqueeze(0).expand_as(preds)
        else:
            raise ValueError(f"label shape {labels.shape} incompatible with preds {preds.shape}")

    valid_mask = labels >= 0
    if not valid_mask.any():
        return valid_logits.new_zeros(())

    return F.binary_cross_entropy_with_logits(
        preds[valid_mask].float(),
        labels[valid_mask].float(),
    )


def compute_topology_loss(
    zi: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    infonce: InfoNCELoss,
    n_hard: int = 32,
) -> torch.Tensor:
    """InfoNCE contrastive loss on true-positive topology embeddings."""
    packed = prepare_infonce_tensors(zi, labels, confidences, n_hard=n_hard)
    if packed is None:
        return zi.new_zeros(())

    zi_anchor, z_pos, z_neg = packed
    return infonce(
        zi_anchor.unsqueeze(0),
        z_pos.unsqueeze(0),
        z_neg.unsqueeze(0),
    )
