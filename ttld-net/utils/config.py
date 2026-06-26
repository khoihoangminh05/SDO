"""YAML configuration loader for TTLD-Net."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Model architecture flags."""

    backbone: str = "yolo26"
    neck: str = "fpn_panet"
    mode: str = "full"
    use_shallow_features: bool = True
    use_soft_nms: bool = True
    heads: list[str] = field(default_factory=lambda: ["tiny_generator", "implicit_topo_sampler", "verification_mlp"])


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 100
    batch_size: int = 32
    optimizer: str = "AdamW"
    lr: float = 1e-4
    scheduler: str = "CosineAnnealing"
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0


@dataclass
class LossConfig:
    """Multi-task loss weights."""

    lambda1: float = 1.0
    lambda2: float = 1.0
    focal_gamma: float = 1.5
    focal_alpha: float = 0.75
    infonce_temperature: float = 0.07
    n_hard_negatives: int = 32


@dataclass
class DataConfig:
    """Dataset paths and loader settings."""

    train_yaml: str = "data/bosch/train.yaml"
    val_yaml: str = "data/bosch/val.yaml"
    ultralytics_yaml: str = "../apps/worker/datasets/bstld.yaml"
    image_size: tuple[int, int] = (720, 1280)
    conf_threshold: float = 0.05
    eval_conf_threshold: float = 0.5
    soft_nms_sigma: float = 0.5
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TTLDConfig:
    """Root configuration object."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    raw: dict[str, Any] = field(default_factory=dict)


def _merge_section(section_cls: type, data: dict[str, Any] | None) -> Any:
    if not data:
        return section_cls()
    valid = {k: v for k, v in data.items() if k in section_cls.__dataclass_fields__}
    return section_cls(**valid)


def load_config(config_path: str | Path) -> TTLDConfig:
    """Load a YAML config file into structured dataclasses."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return TTLDConfig(
        model=_merge_section(ModelConfig, raw.get("model")),
        training=_merge_section(TrainingConfig, raw.get("training")),
        loss=_merge_section(LossConfig, raw.get("loss")),
        data=_merge_section(DataConfig, raw.get("data")),
        raw=raw,
    )


def repo_root() -> Path:
    """Return monorepo root (parent of ttld-net/)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to ttld-net/ then repo root."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()

    ttld_root = Path(__file__).resolve().parents[1]
    for base in (ttld_root, repo_root()):
        resolved = (base / path).resolve()
        if resolved.exists():
            return resolved
    return (ttld_root / path).resolve()
