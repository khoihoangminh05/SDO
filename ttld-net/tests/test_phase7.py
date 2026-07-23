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
    from utils.fast_train import apply_fast_profile

    cfg = load_config(ROOT / "configs" / "m4_full_ttld.yaml")
    apply_fast_profile(cfg)
    assert cfg.training.epochs == 12
    assert cfg.training.max_train_batches == 120
    assert cfg.data.image_size == (480, 640)
    assert cfg.training.use_amp is True

    result = subprocess.run(
        [sys.executable, "scripts/run_ablation.py", "--dry-run", "--configs", "m4_full_ttld"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "train.py" in result.stdout
    assert "test.py" in result.stdout
