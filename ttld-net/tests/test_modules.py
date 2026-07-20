"""Phase gate unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.focal_loss import FocalLoss
from losses.infonce_loss import InfoNCELoss
from models.heads.soft_nms import soft_nms
from models.heads.verification import VerificationMLP
from models.necks.fpn_panet import FeatureProjection


def test_soft_nms_keeps_boxes() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
    scores = torch.tensor([0.9, 0.8])
    kept_boxes, kept_scores = soft_nms(boxes, scores, sigma=0.5)
    assert kept_boxes.shape[0] >= 1
    assert kept_scores.shape[0] >= 1


def test_feature_projection_shape() -> None:
    proj = FeatureProjection(512, 256)
    x = torch.randn(2, 512, 45, 80)
    out = proj(x)
    assert out.shape == (2, 256, 45, 80)


def test_verification_mlp_shape() -> None:
    mlp = VerificationMLP()
    fcand = torch.randn(2, 10, 256)
    zi = torch.randn(2, 10, 256)
    p_logits = mlp(fcand, zi)
    p_valid = torch.sigmoid(p_logits)
    assert p_valid.shape == (2, 10, 1)
    assert (p_valid >= 0).all() and (p_valid <= 1).all()


def test_infonce_easy_less_than_hard() -> None:
    import torch.nn.functional as F

    loss_fn = InfoNCELoss(temperature=0.07)
    zi = F.normalize(torch.randn(2, 5, 256), dim=-1)
    z_pos_easy = zi + 0.01 * torch.randn(2, 5, 256)
    z_pos_hard = F.normalize(torch.randn(2, 5, 256), dim=-1)
    z_neg = F.normalize(torch.randn(2, 5, 3, 256), dim=-1)
    loss_easy = loss_fn(zi, z_pos_easy, z_neg)
    loss_hard = loss_fn(zi, z_pos_hard, z_neg)
    assert loss_easy < loss_hard


def test_focal_loss_runs() -> None:
    loss_fn = FocalLoss(gamma=1.5, alpha=0.75)
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
