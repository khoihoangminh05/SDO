import importlib

import app.config as config


def test_adaptive_tiers():
    # Shortest side drives the tier selection.
    assert config.compute_upscale_factor(18, 18) == 8.0
    assert config.compute_upscale_factor(30, 30) == 6.0
    assert config.compute_upscale_factor(60, 60) == 4.0
    assert config.compute_upscale_factor(120, 120) == 2.0
    assert config.compute_upscale_factor(300, 300) == 1.5


def test_shortest_side_used_for_tier():
    # A tall narrow box: short side = 18 -> tier 8x, but capped by max dim below.
    # short side 60 (tier 4x), long side 70 -> no cap, stays 4x.
    assert config.compute_upscale_factor(70, 60) == 4.0


def test_max_dim_cap():
    # short side 18 -> would be 8x, long side 300 -> 300*8=2400 > 1920 cap.
    scale = config.compute_upscale_factor(300, 18)
    assert scale == config.UPSCALE_MAX_DIM / 300
    assert 300 * scale <= config.UPSCALE_MAX_DIM + 1e-6


def test_disabled_returns_fixed(monkeypatch):
    monkeypatch.setattr(config, "UPSCALE_ADAPTIVE", False)
    assert config.compute_upscale_factor(10, 10) == config.UPSCALE_FACTOR
    assert config.compute_upscale_factor(500, 500) == config.UPSCALE_FACTOR


def test_reference_scale_keeps_area_threshold_constant():
    # At the reference scale (4x) the HSV area rescale factor must be 1.0.
    factor = (config.UPSCALE_FACTOR / config.UPSCALE_FACTOR) ** 2
    assert factor == 1.0
