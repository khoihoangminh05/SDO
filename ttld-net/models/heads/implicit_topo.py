"""Implicit topology sampler via deformable attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.necks.fpn_panet import FeatureProjection


class ImplicitTopologySampler(nn.Module):
    """
    Deformable attention sampler over P4/P5 context features.

    Output topology embedding z_i per candidate (B, N, 256).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, n_points: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_points = n_points
        self.offset_predictor = nn.Linear(d_model, n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_points)
        self.proj = nn.Linear(d_model, d_model)
        self.deform_attn: nn.Module | None = None

        try:
            from mmcv.ops import MultiScaleDeformableAttention

            self.deform_attn = MultiScaleDeformableAttention(
                embed_dims=d_model,
                num_heads=n_heads,
                num_points=n_points,
            )
        except ImportError:  # pragma: no cover
            self.deform_attn = None

    def forward(
        self,
        fcand: torch.Tensor,
        fctx_list: list[torch.Tensor],
        pq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fcand: (B, N, 256) candidate features.
            fctx_list: [(B, 256, H4, W4), (B, 256, H5, W5)] projected context maps.
            pq: (B, N, 2) normalized reference points in [0, 1].

        Returns:
            zi: (B, N, 256) topology embeddings.
        """
        if self.deform_attn is None:
            raise NotImplementedError(
                "Phase 4 (T4.3): install mmcv with CUDA and wire MultiScaleDeformableAttention."
            )
        raise NotImplementedError("Phase 4: complete deformable attention forward pass.")


def extract_reference_points(
    candidates: list[dict],
    image_size: tuple[int, int] = (720, 1280),
) -> torch.Tensor:
    """Convert candidate centers to normalized coordinates (N, 2)."""
    height, width = image_size
    points: list[list[float]] = []
    for candidate in candidates:
        x_center, y_center, _, _ = candidate["bbox"]
        points.append([x_center / width, y_center / height])
    if not points:
        return torch.zeros((0, 2), dtype=torch.float32)
    return torch.tensor(points, dtype=torch.float32)
