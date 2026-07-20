"""Phase 4 gate tests — Implicit Topology Sampler."""

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


def _make_sampler() -> "ImplicitTopologySampler":
    from models.heads.implicit_topo import ImplicitTopologySampler

    return ImplicitTopologySampler(d_model=256, n_heads=8, n_points=4, num_levels=2)


def _make_inputs(
    batch_size: int = 2,
    num_queries: int = 100,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    fcand = torch.randn(batch_size, num_queries, 256, device=device)
    fctx_p4 = torch.randn(batch_size, 256, 45, 80, device=device)
    fctx_p5 = torch.randn(batch_size, 256, 23, 40, device=device)
    pq = torch.rand(batch_size, num_queries, 2, device=device)
    return fcand, [fctx_p4, fctx_p5], pq


def test_topology_sampler_shape_and_finite() -> None:
    """T4.A — output shape (B, N, 256) with no NaN/Inf."""
    sampler = _make_sampler()
    fcand, fctx_list, pq = _make_inputs()

    zi = sampler(fcand, fctx_list, pq)

    assert zi.shape == (2, 100, 256), f"Expected (2, 100, 256), got {zi.shape}"
    assert not torch.isnan(zi).any(), "NaN in topology embedding"
    assert not torch.isinf(zi).any(), "Inf in topology embedding"


def test_topology_sampler_gradient_flow() -> None:
    """T4.B — gradients flow through deformable attention and projection."""
    sampler = _make_sampler()
    fcand, fctx_list, pq = _make_inputs(batch_size=1, num_queries=16)

    zi = sampler(fcand, fctx_list, pq)
    zi.sum().backward()

    assert sampler.proj.weight.grad is not None, "Gradient blocked at proj"
    assert sampler.deform_attn.sampling_offsets.weight.grad is not None, (
        "Gradient blocked at sampling_offsets"
    )
    assert sampler.deform_attn.attention_weights.weight.grad is not None, (
        "Gradient blocked at attention_weights"
    )


def test_topology_sampler_empty_queries() -> None:
    """Edge case — zero candidates returns empty tensor."""
    sampler = _make_sampler()
    fcand = torch.randn(2, 0, 256)
    fctx_p4 = torch.randn(2, 256, 45, 80)
    fctx_p5 = torch.randn(2, 256, 23, 40)
    pq = torch.zeros(2, 0, 2)

    zi = sampler(fcand, [fctx_p4, fctx_p5], pq)
    assert zi.shape == (2, 0, 256)


def test_extract_reference_points() -> None:
    """T4.1 — normalized reference points from candidate bboxes."""
    from models.heads.implicit_topo import (
        build_reference_points_batch,
        extract_reference_points,
    )

    candidates = [
        {"bbox": (640.0, 360.0, 20.0, 20.0)},
        {"bbox": (1280.0, 720.0, 10.0, 10.0)},
    ]
    pq = extract_reference_points(candidates, image_size=(720, 1280))
    assert pq.shape == (2, 2)
    assert torch.allclose(pq[0], torch.tensor([640.0 / 1280, 360.0 / 720]))
    assert torch.allclose(pq[1], torch.tensor([1.0, 1.0]))

    batch_pq = build_reference_points_batch([candidates[:1], candidates[1:]], max_n=3)
    assert batch_pq.shape == (2, 3, 2)
    assert torch.allclose(batch_pq[0, 0], pq[0])
    assert torch.allclose(batch_pq[1, 0], pq[1])


def test_flatten_multiscale_features() -> None:
    """Helper — flattened value length matches spatial shapes."""
    from models.heads.implicit_topo import flatten_multiscale_features

    fctx_p4 = torch.randn(2, 256, 45, 80)
    fctx_p5 = torch.randn(2, 256, 23, 40)
    value, spatial_shapes, level_start_index = flatten_multiscale_features([fctx_p4, fctx_p5])

    assert value.shape == (2, 45 * 80 + 23 * 40, 256)
    assert spatial_shapes.tolist() == [[45, 80], [23, 40]]
    assert level_start_index.tolist() == [0, 45 * 80]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory test")
def test_topology_sampler_memory_smoke() -> None:
    """T4.C — larger N should run without OOM on available GPU."""
    sampler = _make_sampler().cuda()
    fcand, fctx_list, pq = _make_inputs(batch_size=4, num_queries=500, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    zi = sampler(fcand, fctx_list, pq)
    assert zi.shape == (4, 500, 256)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    assert peak_gb < 8.0, f"Unexpected VRAM usage: {peak_gb:.2f} GB"


@pytest.mark.skipif(SKIP_NO_YAML, reason="yolo26_p2.yaml not found")
def test_ttldnet_topology_integration() -> None:
    """Phase 4 — TTLDNet topology mode exposes zi from live stack."""
    from utils.config import load_config

    from models.ttld_net import TTLDNet

    cfg = load_config(ROOT / "configs" / "m3_topology.yaml")
    model = TTLDNet(cfg)
    images = torch.zeros(1, 3, 640, 640)

    outputs = model(images)

    assert "zi" in outputs
    assert outputs["zi"].shape[0] == 1
    assert outputs["zi"].shape[-1] == 256
    assert outputs["zi"].shape[1] == outputs["fcand"].shape[1]
    assert not torch.isnan(outputs["zi"]).any()
