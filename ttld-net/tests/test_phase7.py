"""Phase 7 gate tests — ablation runner smoke."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ablation_configs_exist() -> None:
    names = ["m0_baseline", "m1_shallow", "m2_focal", "m3_topology", "m4_full_ttld"]
    for name in names:
        assert (ROOT / "configs" / f"{name}.yaml").is_file(), f"missing {name}.yaml"


def test_fast_profile_overrides() -> None:
    from utils.config import load_config
    from utils.fast_train import apply_fast_profile, apply_proplus_profile

    cfg = load_config(ROOT / "configs" / "m4_full_ttld.yaml")
    apply_fast_profile(cfg)
    assert cfg.training.epochs == 12
    assert cfg.training.max_train_batches == 120
    assert cfg.data.image_size == (480, 640)
    assert cfg.training.use_amp is False

    cfg2 = load_config(ROOT / "configs" / "m4_full_ttld.yaml")
    apply_proplus_profile(cfg2)
    assert cfg2.training.epochs == 50
    assert cfg2.data.train_subset_ratio == 0.8
    assert cfg2.data.image_size == (640, 1120)
    assert cfg2.training.max_candidates == 100

    result = subprocess.run(
        [sys.executable, "scripts/run_ablation.py", "--dry-run", "--configs", "m4_full_ttld"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "train.py" in result.stdout
    assert "test.py" in result.stdout
    assert "--conf 0.05" in result.stdout


def test_m1_m4_configs_share_core_keys() -> None:
    """Ablation configs stay schema-aligned for professor-facing reproducibility."""
    from utils.config import load_config

    required = {
        "backbone_weights",
        "max_candidates",
        "eval_max_dets",
        "obj_weight",
        "box_weight",
        "soft_nms_sigma",
        "eval_conf_threshold",
    }
    for name in ("m1_shallow", "m2_focal", "m3_topology", "m4_full_ttld"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg.model.backbone_weights
        assert cfg.training.max_candidates > 0
        assert cfg.training.eval_max_dets > 0
        assert cfg.loss.obj_weight > 0
        assert cfg.loss.box_weight > 0
        assert cfg.data.soft_nms_sigma > 0
        assert cfg.data.eval_conf_threshold <= cfg.data.conf_threshold + 1e-9
        _ = required  # documented intent
