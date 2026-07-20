"""Phase 3 gate tests — Semantic Context Branch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

YAML = ROOT.parent / "apps" / "worker" / "models" / "yolo26_p2.yaml"
SKIP_NO_YAML = not YAML.is_file()


def test_feature_projection_shapes() -> None:
    """T3.A — P4/P5 projected to 256-dim with correct spatial size."""
    from models.necks.context_branch import SemanticContextBranch

    b, c4, c5 = 2, 512, 1024
    branch = SemanticContextBranch(c4=c4, c5=c5, c_cand=256, d_model=256)

    p4 = torch.randn(b, c4, 45, 80)
    p5 = torch.randn(b, c5, 23, 40)
    fcand = torch.randn(b, 100, 256)

    fctx_p4, fctx_p5, fcand_proj = branch(p4, p5, fcand)

    assert fctx_p4.shape == (b, 256, 45, 80), f"Fctx_p4 shape sai: {fctx_p4.shape}"
    assert fctx_p5.shape == (b, 256, 23, 40), f"Fctx_p5 shape sai: {fctx_p5.shape}"
    assert fcand_proj is not None and fcand_proj.shape == (b, 100, 256)
    assert not torch.isnan(fctx_p4).any()
    assert not torch.isnan(fctx_p5).any()


def test_feature_projection_gradient_flow() -> None:
    """T3.B — Gradients flow through projection layers."""
    from models.necks.context_branch import SemanticContextBranch

    branch = SemanticContextBranch(c4=128, c5=256, c_cand=256)
    p4 = torch.randn(1, 128, 40, 40, requires_grad=True)
    p5 = torch.randn(1, 256, 20, 20, requires_grad=True)

    fctx_p4, fctx_p5, _ = branch(p4, p5)
    loss = fctx_p4.sum() + fctx_p5.sum()
    loss.backward()

    assert branch.proj_p4.proj.weight.grad is not None, "Gradient blocked at proj_p4"
    assert branch.proj_p5.proj.weight.grad is not None, "Gradient blocked at proj_p5"


@pytest.mark.skipif(SKIP_NO_YAML, reason="yolo26_p2.yaml not found")
def test_ttldnet_context_integration() -> None:
    """T3.3 — TTLDNet forward exposes Fctx from live backbone/neck."""
    from utils.config import load_config

    from models.ttld_net import TTLDNet

    cfg = load_config(ROOT / "configs" / "m1_shallow.yaml")
    model = TTLDNet(cfg)
    images = torch.zeros(1, 3, 640, 640)

    outputs = model(images)

    assert "fctx_p4" in outputs and "fctx_p5" in outputs
    assert outputs["fctx_p4"].shape[1] == 256
    assert outputs["fctx_p5"].shape[1] == 256
    assert outputs["fctx_p4"].shape[2:] == outputs["p4"].shape[2:]
    assert outputs["fctx_p5"].shape[2:] == outputs["p5"].shape[2:]
    assert outputs["fcand_proj"].shape == outputs["fcand"].shape
