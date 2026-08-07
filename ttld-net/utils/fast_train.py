"""Training speed/quality profiles for ablation runs."""

from __future__ import annotations

from utils.config import TTLDConfig


def apply_fast_profile(cfg: TTLDConfig) -> TTLDConfig:
    """
    Shrink train/val cost for limited GPU hours.

    Typical: ~45–90 min per M1–M4 on a mid/high GPU.
    """
    cfg.training.epochs = 12
    cfg.training.batch_size = 4
    cfg.training.val_batch_size = 4
    cfg.training.max_candidates = 64
    cfg.training.eval_max_dets = 50
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
    cfg.data.conf_threshold = 0.05
    cfg.data.eval_conf_threshold = 0.05
    cfg.data.soft_nms_sigma = 0.3

    cfg.loss.n_hard_negatives = 12
    cfg.loss.obj_weight = 2.0
    cfg.loss.box_weight = 3.0
    _sync_raw(cfg, imgsz=640)
    return cfg


def apply_proplus_profile(cfg: TTLDConfig) -> TTLDConfig:
    """
    Last-chance Colab Pro+ profile for M1 (after materialize / pretrained fix).

    Typical: ~3–5 hours for M1 on A100/L4.
    """
    cfg.training.epochs = 50
    cfg.training.batch_size = 4
    cfg.training.val_batch_size = 4
    cfg.training.max_candidates = 100
    cfg.training.eval_max_dets = 100
    cfg.training.use_amp = False
    cfg.training.lr = 1e-4
    cfg.training.warmup_epochs = 3
    cfg.training.aux_loss_warmup_steps = 5
    cfg.training.grad_clip_norm = 0.5
    cfg.training.max_train_batches = 500
    cfg.training.val_max_batches = 40
    cfg.training.eval_every_n_epochs = 5
    cfg.training.log_interval = 50

    # Divisible by 32; near BSTLD aspect ratio.
    cfg.data.image_size = (640, 1120)
    cfg.data.train_subset_ratio = 0.8
    cfg.data.num_workers = 4
    cfg.data.conf_threshold = 0.05
    cfg.data.eval_conf_threshold = 0.05
    cfg.data.soft_nms_sigma = 0.3

    cfg.loss.n_hard_negatives = 24
    cfg.loss.obj_weight = 2.0
    cfg.loss.cls_weight = 1.0
    cfg.loss.box_weight = 3.0
    _sync_raw(cfg, imgsz=1120)
    return cfg


def _sync_raw(cfg: TTLDConfig, imgsz: int) -> None:
    train_raw = cfg.raw.setdefault("training", {})
    train_raw["epochs"] = cfg.training.epochs
    train_raw["batch_size"] = cfg.training.batch_size
    train_raw["imgsz"] = imgsz
    train_raw["max_train_batches"] = cfg.training.max_train_batches
    train_raw["max_candidates"] = cfg.training.max_candidates
    train_raw["eval_max_dets"] = cfg.training.eval_max_dets

    data_raw = cfg.raw.setdefault("data", {})
    data_raw["image_size"] = list(cfg.data.image_size)
    data_raw["train_subset_ratio"] = cfg.data.train_subset_ratio
    data_raw["conf_threshold"] = cfg.data.conf_threshold
    data_raw["eval_conf_threshold"] = cfg.data.eval_conf_threshold
    data_raw["soft_nms_sigma"] = cfg.data.soft_nms_sigma

    loss_raw = cfg.raw.setdefault("loss", {})
    loss_raw["obj_weight"] = cfg.loss.obj_weight
    loss_raw["cls_weight"] = cfg.loss.cls_weight
    loss_raw["box_weight"] = cfg.loss.box_weight

    model_raw = cfg.raw.setdefault("model", {})
    model_raw.setdefault("backbone_weights", "yolo26n.pt")
