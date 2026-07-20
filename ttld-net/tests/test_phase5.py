"""Phase 5 gate tests — Verification MLP + InfoNCE contrastive learning."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

YAML = ROOT.parent / "apps" / "worker" / "models" / "yolo26_p2.yaml"
SKIP_NO_YAML = not YAML.is_file()


def test_verification_mlp_shape() -> None:
    """T5.A — VerificationMLP output shape and valid probability range."""
    from models.heads.verification import VerificationMLP

    mlp = VerificationMLP()
    fcand = torch.randn(2, 10, 256)
    zi = torch.randn(2, 10, 256)

    p_logits = mlp(fcand, zi)
    p_valid = torch.sigmoid(p_logits)
    assert p_valid.shape == (2, 10, 1), f"Shape sai: {p_valid.shape}"
    assert (p_valid >= 0).all() and (p_valid <= 1).all()


def test_infonce_convergence() -> None:
    """T5.B — InfoNCE lower when positive pairs are closer to anchors."""
    from losses.infonce_loss import InfoNCELoss

    loss_fn = InfoNCELoss(temperature=0.07)
    zi = F.normalize(torch.randn(2, 10, 256), dim=-1)
    z_pos_easy = zi + 0.01 * torch.randn(2, 10, 256)
    z_pos_hard = F.normalize(torch.randn(2, 10, 256), dim=-1)
    z_neg = F.normalize(torch.randn(2, 10, 5, 256), dim=-1)

    loss_easy = loss_fn(zi, z_pos_easy, z_neg)
    loss_hard = loss_fn(zi, z_pos_hard, z_neg)
    assert loss_easy < loss_hard, f"InfoNCE failed: {loss_easy:.4f} >= {loss_hard:.4f}"


def test_match_candidates_to_gt() -> None:
    """T5.3 — IoU matching labels TP/FP correctly."""
    from models.heads.verification import label_candidates_batch, match_candidates_to_gt

    gt = torch.tensor([[100.0, 100.0, 20.0, 20.0, 2.0]])
    tp_cand = [{"bbox": (100.0, 100.0, 18.0, 18.0), "confidence": 0.9}]
    fp_cand = [{"bbox": (400.0, 400.0, 10.0, 10.0), "confidence": 0.8}]

    tp_labels = match_candidates_to_gt(tp_cand, gt)
    fp_labels = match_candidates_to_gt(fp_cand, gt)
    assert tp_labels[0] == 1.0
    assert fp_labels[0] == 0.0

    candidates = [[*tp_cand, *fp_cand]]
    targets = [gt]
    labels, confidences = label_candidates_batch(candidates, targets, max_n=4)
    assert labels[0, 0] == 1.0
    assert labels[0, 1] == 0.0
    assert labels[0, 2] == -1.0  # padding
    assert confidences[0, 0] == pytest.approx(0.9)
    assert candidates[0][1]["is_false_positive"] is True


def test_mine_hard_negatives() -> None:
    """T5.3 — Hard negatives are highest-confidence false positives."""
    from models.heads.verification import mine_hard_negatives

    candidates = [
        {"confidence": 0.5, "is_false_positive": True},
        {"confidence": 0.95, "is_false_positive": True},
        {"confidence": 0.7, "is_false_positive": True},
        {"confidence": 0.99, "is_false_positive": False},
    ]
    hard = mine_hard_negatives(candidates, n_hard=2)
    assert len(hard) == 2
    assert hard[0]["confidence"] == 0.95
    assert hard[1]["confidence"] == 0.7


def test_prepare_infonce_tensors() -> None:
    """T5.4 — Positive pairs and hard negatives packed for InfoNCE."""
    from models.heads.verification import prepare_infonce_tensors

    zi = torch.randn(2, 4, 256)
    labels = torch.tensor(
        [
            [1.0, 1.0, 0.0, -1.0],
            [1.0, 0.0, 0.0, -1.0],
        ]
    )
    confidences = torch.tensor(
        [
            [0.9, 0.8, 0.95, 0.0],
            [0.7, 0.6, 0.5, 0.0],
        ]
    )

    packed = prepare_infonce_tensors(zi, labels, confidences, n_hard=2)
    assert packed is not None
    anchors, positives, negatives = packed
    assert anchors.shape[0] == 3  # 2 TP in img0 + 1 TP in img1
    assert positives.shape == anchors.shape
    assert negatives.shape == (3, 2, 256)


def test_verification_bce_loss_ignores_padding() -> None:
    """BCE loss skips padded candidate slots (label == -1)."""
    from models.heads.verification import verification_bce_loss

    p_logits = torch.tensor([[[2.0], [-2.0], [0.0]]])
    labels = torch.tensor([[1.0, 0.0, -1.0]])
    loss = verification_bce_loss(p_logits, labels)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_compute_topology_loss_runs() -> None:
    """Topology loss is finite when true positives exist."""
    from losses.infonce_loss import InfoNCELoss
    from models.heads.verification import compute_topology_loss

    zi = torch.randn(1, 3, 256)
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    confidences = torch.tensor([[0.9, 0.95, 0.5]])
    loss = compute_topology_loss(zi, labels, confidences, InfoNCELoss(), n_hard=2)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


@pytest.mark.skipif(SKIP_NO_YAML, reason="yolo26_p2.yaml not found")
def test_ttldnet_full_loss_integration() -> None:
    """Phase 5 — full mode forward + compute_losses (verify + topology)."""
    from losses.infonce_loss import InfoNCELoss
    from utils.config import load_config

    from models.ttld_net import TTLDNet

    cfg = load_config(ROOT / "configs" / "m4_full_ttld.yaml")
    model = TTLDNet(cfg)
    images = torch.zeros(1, 3, 640, 640)
    targets = [torch.tensor([[320.0, 320.0, 30.0, 30.0, 2.0]])]

    outputs = model(images)
    assert "zi" in outputs
    assert "p_valid" in outputs
    assert outputs["p_valid"].shape[-1] == 1

    losses = model.compute_losses(
        outputs,
        targets,
        loss_fns={"infonce": InfoNCELoss(temperature=cfg.loss.infonce_temperature)},
    )
    assert "verify" in losses
    assert "topology" in losses
    assert "total" in losses
    assert losses["total"].ndim == 0
    assert not torch.isnan(losses["total"])


@pytest.mark.skipif(SKIP_NO_YAML, reason="yolo26_p2.yaml not found")
def test_ttldnet_topology_mode_skips_verifier() -> None:
    """M3 topology mode exposes zi but not p_valid."""
    from utils.config import load_config

    from models.ttld_net import TTLDNet

    cfg = load_config(ROOT / "configs" / "m3_topology.yaml")
    model = TTLDNet(cfg)
    outputs = model(torch.zeros(1, 3, 640, 640))

    assert "zi" in outputs
    assert "p_valid" not in outputs
