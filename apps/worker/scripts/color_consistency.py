"""
Color Consistency Loss for Traffic Light Detection.

Novel research contribution:
  Standard detectors learn geometry + class from labeled boxes, but ignore the
  physical color prior of traffic lights (red/yellow/green are emitted light).
  This module adds an auxiliary loss L_color that penalizes predictions where
  the GT bounding box's HSV color distribution does not match the class label.

  L_total = L_detect  +  lambda_color * L_color
  L_color = weighted_L1(mean_HSV(crop), target_HSV[class])

Key design decisions:
  - HSV space (not RGB): hue directly encodes traffic light color, robust to
    brightness/exposure variation (value channel can vary widely).
  - Weighted dimensions: hue is most discriminative for red/yellow/green;
    value is most discriminative for "off" (dark) class.
  - Warmup skip: skip color loss for first WARMUP_EPOCHS to let the detector
    stabilize before adding auxiliary signal.
  - Min box area filter: tiny crops (<16px²) have unreliable color stats.

Usage:
  from scripts.color_consistency import ColorConsistencyTrainer
  trainer = ColorConsistencyTrainer(overrides={...})
  trainer.train()

Paper citation target:
  "Physics-Guided Color Consistency Loss for Tiny Traffic Light Detection"
"""

import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer


# ── HSV target centers per class ─────────────────────────────────────────────
# Empirically calibrated on BSTLD dataset (normalized to [0, 1]).
# H: hue (0=red, 0.12=yellow, 0.36=green)
# S: saturation (high for lit LED, low for off)
# V: value/brightness (high for lit, low for off)
_CLASS_HSV = torch.tensor([
    [0.00, 0.80, 0.78],   # 0: red    — hue near 0, saturated, bright
    [0.12, 0.85, 0.88],   # 1: yellow — hue ~0.12, saturated, very bright
    [0.36, 0.75, 0.72],   # 2: green  — hue ~0.36
    [0.00, 0.08, 0.22],   # 3: off    — low saturation, dark
], dtype=torch.float32)

# Per-dimension loss weights [H, S, V]
# Hue 2× for color classes; value 2× for "off" class.
_HSV_W_COLOR = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float32)
_HSV_W_OFF   = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float32)

# Red wraps around hue=0/1, needs special handling.
_RED_HUE_ALT = 1.0   # equivalent hue for red on the other side of the circle


# ── Differentiable RGB → HSV ─────────────────────────────────────────────────

def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert [*, 3, H, W] RGB tensor in [0, 1] to HSV in [0, 1].
    Differentiable (uses clamp/where instead of if-branches on scalar paths).
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    max_c, _ = rgb.max(dim=1)
    min_c, _ = rgb.min(dim=1)
    diff = (max_c - min_c).clamp(min=1e-7)

    # Hue in [0, 6)
    h = torch.zeros_like(r)
    mask_r = (max_c == r)
    mask_g = (max_c == g) & ~mask_r
    mask_b = (max_c == b) & ~mask_r & ~mask_g

    h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6.0
    h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2.0
    h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4.0
    h = h / 6.0  # normalize to [0, 1]

    s = torch.where(max_c > 0, diff / max_c, torch.zeros_like(max_c))
    v = max_c

    return torch.stack([h, s, v], dim=1)   # [*, 3, H, W]


def _mean_hsv_of_crop(img_batch: torch.Tensor,
                      b: int, x1: int, y1: int, x2: int, y2: int
                      ) -> torch.Tensor:
    """Return mean HSV [3] for a single crop region."""
    crop = img_batch[b:b+1, :, y1:y2, x1:x2]   # [1, 3, h, w]
    hsv  = rgb_to_hsv(crop)                       # [1, 3, h, w]
    return hsv.mean(dim=[0, 2, 3])                # [3]


def color_loss_for_batch(batch: dict,
                         lambda_color: float = 0.08,
                         min_box_area: int = 16,
                         warmup_done: bool = True
                         ) -> torch.Tensor:
    """
    Compute Color Consistency Loss for a training batch.

    Args:
        batch:          Ultralytics batch dict with keys:
                          'img'       [B, 3, H, W] float32 in [0, 1]
                          'cls'       [N, 1] int64
                          'bboxes'    [N, 4] float32 cx cy w h (normalized)
                          'batch_idx' [N, 1] int64
        lambda_color:   Weight of the color loss (paper uses 0.08).
        min_box_area:   Skip crops smaller than this many pixels.
        warmup_done:    False during warmup → return zero loss.

    Returns:
        Scalar loss tensor (gradient-attached).
    """
    if not warmup_done:
        return torch.tensor(0.0, device=batch['img'].device)

    imgs      = batch['img']                        # [B, 3, H, W]
    cls       = batch['cls'].long().view(-1)        # [N]
    bboxes    = batch['bboxes']                     # [N, 4]
    batch_idx = batch['batch_idx'].long().view(-1)  # [N]

    device = imgs.device
    _, _, H, W = imgs.shape

    hsv_centers = _CLASS_HSV.to(device)
    w_color     = _HSV_W_COLOR.to(device)
    w_off       = _HSV_W_OFF.to(device)

    losses = []
    for i in range(len(cls)):
        c  = cls[i].item()
        bi = batch_idx[i].item()
        cx, cy, bw, bh = bboxes[i].tolist()

        x1 = max(0, int((cx - bw / 2) * W))
        y1 = max(0, int((cy - bh / 2) * H))
        x2 = min(W, int((cx + bw / 2) * W))
        y2 = min(H, int((cy + bh / 2) * H))

        if (x2 - x1) * (y2 - y1) < min_box_area:
            continue

        mean_hsv = _mean_hsv_of_crop(imgs, bi, x1, y1, x2, y2)  # [3]
        target   = hsv_centers[c]                                  # [3]
        weights  = w_off if c == 3 else w_color

        # Red wraps: use min circular distance for hue channel
        if c == 0:  # red
            hue_diff = torch.min(
                (mean_hsv[0] - target[0]).abs(),
                (mean_hsv[0] - _RED_HUE_ALT).abs()
            )
            sv_diff  = (mean_hsv[1:] - target[1:]).abs()
            per_dim  = torch.cat([hue_diff.unsqueeze(0), sv_diff])
        else:
            per_dim  = (mean_hsv - target).abs()

        losses.append((weights * per_dim).mean())

    if not losses:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean() * lambda_color


# ── Custom Ultralytics Trainer ────────────────────────────────────────────────

class ColorConsistencyTrainer(DetectionTrainer):
    """
    DetectionTrainer subclass that injects Color Consistency Loss.

    DO NOT pass custom keys in overrides — Ultralytics rejects unknown keys.
    Instead, set instance attributes AFTER construction, BEFORE .train():

        trainer = ColorConsistencyTrainer(overrides={...YOLO keys only...})
        trainer._lambda_color = 0.08   # color loss weight
        trainer._color_warmup = 15     # epoch to start color loss
        trainer._min_box_area = 16     # skip crop < N pixels²
        trainer._pretrained   = 'yolo26s.pt'
        trainer.train()

    Or use the helper function make_trainer() below.
    """

    # Defaults — set as class attributes, overridden per-instance after __init__
    _lambda_color: float = 0.08
    _color_warmup: int   = 15
    _min_box_area: int   = 16
    _pretrained:   str   = None

    def setup_model(self):
        """Build model from yaml then transfer pretrained backbone weights."""
        super().setup_model()
        if not self._pretrained:
            return
        import torch
        from ultralytics.utils.torch_utils import intersect_dicts
        from ultralytics import YOLO as _YOLO
        print(f'[ColorConsistencyTrainer] Transferring weights from: {self._pretrained}')
        # Load source model (auto-downloads if needed), get its nn.Module
        src_model = _YOLO(self._pretrained).model
        src_sd    = src_model.float().state_dict()
        dst_sd    = self.model.state_dict()
        matched   = intersect_dicts(src_sd, dst_sd)
        self.model.load_state_dict(matched, strict=False)
        print(f'  Transferred {len(matched)}/{len(dst_sd)} parameter groups')

    def criterion(self, preds, batch):
        loss, loss_items = super().criterion(preds, batch)

        warmup_done = (self.epoch >= self._color_warmup)
        c_loss = color_loss_for_batch(
            batch,
            lambda_color=self._lambda_color,
            min_box_area=self._min_box_area,
            warmup_done=warmup_done,
        )
        loss = loss + c_loss
        loss_items = torch.cat([loss_items, c_loss.detach().unsqueeze(0)])
        return loss, loss_items


def make_trainer(
    model_yaml: str,
    data_yaml:  str,
    pretrained: str  = 'yolo26s.pt',
    imgsz:      int  = 1280,
    epochs:     int  = 150,
    patience:   int  = 40,
    batch:      int  = -1,
    project:    str  = 'runs',
    name:       str  = 'yolo26p2_color',
    lambda_color: float = 0.08,
    color_warmup: int   = 15,
    min_box_area: int   = 16,
    **extra_overrides,
) -> ColorConsistencyTrainer:
    """
    Helper that constructs a ColorConsistencyTrainer without touching
    Ultralytics cfg validation.

    Usage:
        trainer = make_trainer(
            model_yaml = '/content/bstld/yolo26_p2.yaml',
            data_yaml  = '/content/bstld/bstld.yaml',
            pretrained = 'yolo26s.pt',
            project    = '/content/drive/MyDrive/SDO_train/runs',
            name       = 'yolo26p2_color_bstld_1280',
        )
        trainer.train()
    """
    # Only standard YOLO keys here — no custom keys
    yolo_overrides = dict(
        model    = model_yaml,
        data     = data_yaml,
        imgsz    = imgsz,
        epochs   = epochs,
        patience = patience,
        batch    = batch,
        amp      = True,
        cache    = 'disk',
        workers  = 4,
        mosaic   = 1.0,
        close_mosaic = 10,
        scale    = 0.2,
        hsv_h    = 0.01,
        hsv_s    = 0.7,
        hsv_v    = 0.4,
        flipud   = 0.0,
        fliplr   = 0.5,
        copy_paste = 0.1,
        project  = project,
        name     = name,
        exist_ok = True,
        save     = True,
        **extra_overrides,
    )

    trainer = ColorConsistencyTrainer(overrides=yolo_overrides)

    # Set custom attrs AFTER construction (bypasses Ultralytics validation)
    trainer._lambda_color = lambda_color
    trainer._color_warmup = color_warmup
    trainer._min_box_area = min_box_area
    trainer._pretrained   = pretrained

    return trainer
