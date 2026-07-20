"""Implicit topology sampler via deformable attention."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    CPU-compatible multi-scale deformable attention (ported from mmcv).

    Args:
        value: (bs, num_keys, num_heads, head_dim)
        value_spatial_shapes: (num_levels, 2) with (h, w) per level
        sampling_locations: (bs, num_queries, num_heads, num_levels, num_points, 2)
        attention_weights: (bs, num_queries, num_heads, num_levels, num_points)
    """
    bs, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split(
        [int(h.item()) * int(w.item()) for h, w in value_spatial_shapes],
        dim=1,
    )
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list: list[torch.Tensor] = []

    for level, (height, width) in enumerate(value_spatial_shapes):
        h_, w_ = int(height.item()), int(width.item())
        value_l = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, head_dim, h_, w_)
        )
        sampling_grid_l = (
            sampling_grids[:, :, :, level]
            .transpose(1, 2)
            .flatten(0, 1)
        )
        sampling_value_l = F.grid_sample(
            value_l,
            sampling_grid_l,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
    ).sum(-1)
    output = output.view(bs, num_heads * head_dim, num_queries)
    return output.transpose(1, 2).contiguous()


class _PyTorchMSDeformAttn(nn.Module):
    """Pure-PyTorch deformable attention with mmcv-compatible batch_first API."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 2,
        num_points: int = 4,
    ) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError("embed_dims must be divisible by num_heads")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = (
            grid_init / grid_init.abs().max(dim=-1, keepdim=True).values
        ).view(self.num_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.num_levels, self.num_points, 1)
        for point_idx in range(self.num_points):
            grid_init[:, :, point_idx, :] *= point_idx + 1
        self.sampling_offsets.bias = nn.Parameter(grid_init.reshape(-1))

        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del key, level_start_index, kwargs

        if value is None:
            value = query
        if identity is None:
            identity = query
        if reference_points is None or spatial_shapes is None:
            raise ValueError("reference_points and spatial_shapes are required")

        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        attn = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attn = attn.softmax(dim=-1).view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        offset_normalizer = torch.stack(
            [spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )

        output = _multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attn
        )
        return self.output_proj(output) + identity


def flatten_multiscale_features(
    fctx_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten multi-scale context maps for deformable attention.

    Returns:
        value: (B, num_keys, C)
        spatial_shapes: (num_levels, 2) as (h, w)
        level_start_index: (num_levels,)
    """
    if not fctx_list:
        raise ValueError("fctx_list must contain at least one feature map")

    flattened: list[torch.Tensor] = []
    spatial_shapes: list[list[int]] = []
    for feat in fctx_list:
        _, _, height, width = feat.shape
        spatial_shapes.append([height, width])
        flattened.append(feat.flatten(2).transpose(1, 2))

    value = torch.cat(flattened, dim=1)
    device = value.device
    spatial_shapes_t = torch.tensor(spatial_shapes, dtype=torch.long, device=device)
    level_start_index = torch.zeros(len(fctx_list), dtype=torch.long, device=device)
    for level in range(1, len(fctx_list)):
        prev_h, prev_w = spatial_shapes[level - 1]
        level_start_index[level] = level_start_index[level - 1] + prev_h * prev_w
    return value, spatial_shapes_t, level_start_index


def expand_reference_points(pq: torch.Tensor, num_levels: int) -> torch.Tensor:
    """Expand (B, N, 2) normalized points to (B, N, num_levels, 2)."""
    return pq.unsqueeze(2).expand(-1, -1, num_levels, -1).contiguous()


class ImplicitTopologySampler(nn.Module):
    """
    Deformable attention sampler over P4/P5 context features.

    Output topology embedding z_i per candidate (B, N, 256).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_points: int = 4,
        num_levels: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_points = n_points
        self.num_levels = num_levels
        self.proj = nn.Linear(d_model, d_model)

        deform_attn: nn.Module | None = None
        try:
            from mmcv.ops import MultiScaleDeformableAttention

            deform_attn = MultiScaleDeformableAttention(
                embed_dims=d_model,
                num_heads=n_heads,
                num_levels=num_levels,
                num_points=n_points,
                batch_first=True,
                dropout=0.0,
            )
        except ImportError:
            deform_attn = _PyTorchMSDeformAttn(
                embed_dims=d_model,
                num_heads=n_heads,
                num_levels=num_levels,
                num_points=n_points,
            )

        self.deform_attn = deform_attn

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
        if fcand.ndim != 3:
            raise ValueError(f"fcand must be (B, N, C), got {fcand.shape}")

        batch_size, num_queries, channels = fcand.shape
        if channels != self.d_model:
            raise ValueError(
                f"fcand channels {channels} != d_model {self.d_model}"
            )
        if pq.shape[:2] != (batch_size, num_queries):
            raise ValueError(
                f"pq shape {pq.shape} must match fcand batch/query dims "
                f"({batch_size}, {num_queries})"
            )
        if len(fctx_list) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} context levels, got {len(fctx_list)}"
            )

        if num_queries == 0:
            return fcand.new_zeros(batch_size, 0, self.d_model)

        pq = pq.to(device=fcand.device, dtype=fcand.dtype)
        value, spatial_shapes, level_start_index = flatten_multiscale_features(fctx_list)
        reference_points = expand_reference_points(pq, self.num_levels)

        attended = self.deform_attn(
            query=fcand,
            key=value,
            value=value,
            identity=fcand,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        return self.proj(attended)


def extract_reference_points(
    candidates: list[dict[str, Any]],
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


def build_reference_points_batch(
    candidates: list[list[dict[str, Any]]],
    image_size: tuple[int, int] = (720, 1280),
    max_n: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Build padded (B, N, 2) normalized reference points aligned with fcand.

    Padding slots are zeros (invalid candidates padded by TinyGenerator).
    """
    batch_points = [extract_reference_points(cands, image_size) for cands in candidates]
    if max_n is None:
        max_n = max((pts.shape[0] for pts in batch_points), default=0)
    max_n = max(max_n, 1)

    result = torch.zeros(len(batch_points), max_n, 2, dtype=torch.float32)
    for batch_idx, pts in enumerate(batch_points):
        n = min(pts.shape[0], max_n)
        if n > 0:
            result[batch_idx, :n] = pts[:n]

    if device is not None or dtype is not None:
        result = result.to(device=device, dtype=dtype or result.dtype)
    return result
