"""Phase 2 gate tests — backbone, neck, TinyGenerator."""

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


@pytest.mark.skipif(SKIP_NO_YAML, reason="yolo26_p2.yaml not found")
def test_backbone_p1_p5_shapes() -> None:
    """T2.1 — Backbone emits five pyramid levels."""
    from models.backbones.yolo26 import YOLO26Backbone

    backbone = YOLO26Backbone(yaml_path=YAML)
    dummy = torch.zeros(1, 3, 640, 640)
    p1, p2, p3, p4, p5 = backbone(dummy)

    assert p1.shape[2:] == (320, 320)
    assert p2.shape[2:] == (160, 160)
    assert p3.shape[2:] == (80, 80)
    assert p4.shape[2:] == (40, 40)
    assert p5.shape[2:] == (20, 20)
    dims = backbone.channel_dims((640, 640))
    assert dims["p1"] == p1.shape[1]
    assert dims["p5"] == p5.shape[1]


def test_fpn_panet_forward() -> None:
    """T2.2 — Neck unifies channels to 256."""
    from models.necks.fpn_panet import FPNPANet

    neck = FPNPANet(in_channels=(16, 64, 128, 128, 256), out_channels=256)
    feats = (
        torch.randn(2, 16, 160, 160),
        torch.randn(2, 64, 80, 80),
        torch.randn(2, 128, 40, 40),
        torch.randn(2, 128, 20, 20),
        torch.randn(2, 256, 10, 10),
    )
    o1, o2, o3, o4, o5 = neck(feats)
    for out, spatial in zip((o1, o2, o3, o4, o5), feats):
        assert out.shape[0] == 2
        assert out.shape[1] == 256
        assert out.shape[2:] == spatial.shape[2:]


def test_generator_forward() -> None:
    """T2.A — TinyGenerator produces candidates + fcand."""
    from models.heads.tiny_generator import TinyGenerator

    gen = TinyGenerator(
        in_channels=(256, 256, 256),
        strides=(2, 4, 8),
        conf_threshold=0.01,
        max_candidates=500,
    )
    b = 2
    p1 = torch.randn(b, 256, 80, 80)
    p2 = torch.randn(b, 256, 40, 40)
    p3 = torch.randn(b, 256, 20, 20)
    candidates, fcand = gen(p1, p2, p3)

    assert len(candidates) == b
    n = fcand.shape[1]
    assert fcand.shape == (b, n, 256)
    assert not torch.isnan(fcand).any()
    # With low conf + random weights, expect a non-trivial candidate pool
    assert n >= 1
    assert sum(len(c) for c in candidates) >= 1


@pytest.mark.skipif(SKIP_NO_YAML, reason="yolo26_p2.yaml not found")
def test_phase2_stack_smoke() -> None:
    """Backbone → Neck → Generator end-to-end smoke."""
    from models.backbones.yolo26 import YOLO26Backbone
    from models.heads.tiny_generator import TinyGenerator
    from models.necks.fpn_panet import FPNPANet

    backbone = YOLO26Backbone(yaml_path=YAML)
    dims = backbone.channel_dims((640, 640))
    neck = FPNPANet(
        in_channels=(dims["p1"], dims["p2"], dims["p3"], dims["p4"], dims["p5"]),
        out_channels=256,
    )
    gen = TinyGenerator(conf_threshold=0.01, max_candidates=300)

    x = torch.zeros(1, 3, 640, 640)
    pyramids = backbone(x)
    p1, p2, p3, p4, p5 = neck(pyramids)
    candidates, fcand = gen(p1, p2, p3)

    assert p4.shape[1] == 256 and p5.shape[1] == 256
    assert fcand.shape[-1] == 256
    assert isinstance(candidates[0], list)
