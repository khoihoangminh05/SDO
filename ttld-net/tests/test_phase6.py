"""Phase 6 gate tests — end-to-end training."""

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


def _ttld_for_test(config_name: str = "m4_full_ttld.yaml"):
    from utils.config import load_config
    from models.ttld_net import TTLDNet

    cfg = load_config(ROOT / "configs" / config_name)
    cfg.model.backbone_weights = None  # unit tests: no pretrained download
    return cfg, TTLDNet(cfg)


def test_e2e_forward_full_mode() -> None:
    """T6.A — full TTLD-Net forward exposes zi + p_valid without NaN."""
    _, model = _ttld_for_test()
    images = torch.randn(2, 3, 640, 640)

    outputs = model(images)

    assert "zi" in outputs and "p_valid" in outputs
    assert "raw_outputs" in outputs
    assert outputs["p_valid"].shape[-1] == 1
    assert not torch.isnan(outputs["p_valid"]).any()
    assert not torch.isnan(outputs["zi"]).any()


def test_compute_losses_includes_det() -> None:
    """L_total includes L_det + auxiliary losses."""
    from losses.infonce_loss import InfoNCELoss

    _, model = _ttld_for_test()
    images = torch.randn(2, 3, 640, 640)
    targets = [
        torch.tensor([[320.0, 320.0, 40.0, 40.0, 2.0]]),
        torch.tensor([[160.0, 160.0, 20.0, 20.0, 0.0]]),
    ]

    outputs = model(images)
    losses = model.compute_losses(
        outputs,
        targets,
        loss_fns={"infonce": InfoNCELoss()},
    )

    assert "det" in losses
    assert "topology" in losses
    assert "verify" in losses
    assert "total" in losses
    assert losses["total"].ndim == 0
    assert float(losses["total"]) > 0
    assert not torch.isnan(losses["total"])


def test_detection_loss_runs() -> None:
    """DetectionLoss produces finite scalar from raw head outputs."""
    from losses.detection_loss import DetectionLoss

    raw = [
        {
            "obj": torch.randn(2, 1, 80, 80),
            "cls": torch.randn(2, 4, 80, 80),
            "box": torch.randn(2, 4, 80, 80),
            "stride": 8,
        }
    ]
    targets = [
        torch.tensor([[320.0, 320.0, 30.0, 30.0, 2.0]]),
        torch.tensor([[100.0, 100.0, 15.0, 15.0, 1.0]]),
    ]
    loss = DetectionLoss()(raw, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_detection_loss_large_grid_finite() -> None:
    """Sampled objectness loss stays finite on P1-sized grids."""
    from losses.detection_loss import DetectionLoss

    raw = [
        {
            "obj": torch.randn(1, 1, 360, 640),
            "cls": torch.randn(1, 4, 360, 640),
            "box": torch.randn(1, 4, 360, 640),
            "stride": 2,
        }
    ]
    targets = [torch.tensor([[640.0, 360.0, 40.0, 40.0, 2.0]])]
    loss = DetectionLoss(max_obj_negatives=2048)(raw, targets)
    assert torch.isfinite(loss)


def test_training_step_backward() -> None:
    """T6.B smoke — one optimizer step AFTER materialize (neck/backbone covered)."""
    from losses.infonce_loss import InfoNCELoss

    _, model = _ttld_for_test()
    model.materialize(torch.device("cpu"), (640, 640))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    orphans = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in opt_ids]
    assert not orphans, f"orphan params: {orphans[:8]}"

    images = torch.randn(1, 3, 640, 640)
    targets = [torch.tensor([[320.0, 320.0, 30.0, 30.0, 2.0]])]

    outputs = model(images)
    losses = model.compute_losses(outputs, targets, loss_fns={"infonce": InfoNCELoss()})

    assert losses["total"].requires_grad
    optimizer.zero_grad()
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    assert torch.isfinite(losses["total"])


def test_materialize_before_optim_covers_neck() -> None:
    """Regression: lazy neck must exist in optimizer param list."""
    _, model = _ttld_for_test("m1_shallow.yaml")
    # BUG pattern (old): optim before first forward → neck placeholder only
    bad_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    bad_ids = {id(p) for g in bad_opt.param_groups for p in g["params"]}
    model.materialize(torch.device("cpu"), (640, 640))
    neck_orphans = [
        n for n, p in model.named_parameters()
        if n.startswith("neck.") and p.requires_grad and id(p) not in bad_ids
    ]
    assert neck_orphans, "expected neck to be orphaned if optim created too early"

    good_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    good_ids = {id(p) for g in good_opt.param_groups for p in g["params"]}
    still_orphan = [
        n for n, p in model.named_parameters()
        if p.requires_grad and id(p) not in good_ids
    ]
    assert not still_orphan
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 1_000_000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_amp_training_loss_finite() -> None:
    """AMP forward + fp32 loss must stay finite on GPU."""
    from losses.infonce_loss import InfoNCELoss

    _, model = _ttld_for_test()
    model = model.cuda()
    model.materialize(torch.device("cuda"), (640, 640))
    images = torch.randn(1, 3, 640, 640, device="cuda")
    targets = [torch.tensor([[320.0, 320.0, 30.0, 30.0, 2.0]], device="cuda")]

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(images)
    with torch.autocast(device_type="cuda", enabled=False):
        losses = model.compute_losses(outputs, targets, loss_fns={"infonce": InfoNCELoss()})

    assert torch.isfinite(losses["total"]).all()
    assert not torch.isnan(outputs["zi"]).any()


@pytest.mark.skipif(SKIP_NO_YAML, reason="yolo26_p2.yaml not found")
def test_evaluate_model_smoke() -> None:
    """Metrics pipeline runs on one synthetic batch."""
    from utils.metrics import evaluate_model

    _, model = _ttld_for_test("m3_topology.yaml")
    model.materialize(torch.device("cpu"), (640, 640))

    class _FakeLoader:
        def __iter__(self):
            images = torch.zeros(1, 3, 640, 640)
            targets = [torch.tensor([[320.0, 320.0, 30.0, 30.0, 2.0]])]
            yield images, targets

    metrics = evaluate_model(model, _FakeLoader(), torch.device("cpu"), max_batches=1)
    assert "recall" in metrics
    assert "precision" in metrics
    assert "ap50" in metrics


def test_average_precision_perfect_ranking() -> None:
    """True AP50 is 1.0 when all GT are hit first."""
    from utils.metrics import aggregate_matches, average_precision

    matches = [(0.9, 1), (0.8, 1), (0.1, 0), (0.05, 0)]
    assert average_precision(matches, num_gt=2) == pytest.approx(1.0, abs=1e-6)
    metrics = aggregate_matches(matches, total_gt=2, operating_conf=0.05)
    assert metrics["ap50"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] < 1.0


def test_average_precision_zero_when_no_tp() -> None:
    from utils.metrics import average_precision

    assert average_precision([(0.9, 0), (0.8, 0)], num_gt=3) == 0.0
    assert average_precision([], num_gt=3) == 0.0
