"""Fast ablation training profile — trades accuracy for wall-clock time."""

from __future__ import annotations

from utils.config import TTLDConfig


def apply_fast_profile(cfg: TTLDConfig) -> TTLDConfig:
    """
    Shrink train/val cost for Phase-7 ablation on limited GPU hours.

    Typical effect on RTX 4090: ~45–90 min per M1–M4 variant (vs many days).
    """
    # Speed from resolution/subset/steps — keep AMP off (fp16 grads often Inf on this stack).
    cfg.training.epochs = 12
    cfg.training.batch_size = 4
    cfg.training.val_batch_size = 4
    cfg.training.max_candidates = 64
    cfg.training.use_amp = False
    cfg.training.lr = 8e-5
    cfg.training.warmup_epochs = 1
    cfg.training.aux_loss_warmup_steps = 2
    cfg.training.grad_clip_norm = 0.5
    cfg.training.max_train_batches = 120
    cfg.training.val_max_batches = 20
    cfg.training.eval_every_n_epochs = 4
    cfg.training.log_interval = 20

    cfg.data.image_size = (480, 640)
    cfg.data.train_subset_ratio = 0.25
    cfg.data.num_workers = 4

    cfg.loss.n_hard_negatives = 12

    train_raw = cfg.raw.setdefault("training", {})
    train_raw["epochs"] = cfg.training.epochs
    train_raw["batch_size"] = cfg.training.batch_size
    train_raw["imgsz"] = 640
    train_raw["max_train_batches"] = cfg.training.max_train_batches

    data_raw = cfg.raw.setdefault("data", {})
    data_raw["image_size"] = list(cfg.data.image_size)
    data_raw["train_subset_ratio"] = cfg.data.train_subset_ratio

    return cfg
