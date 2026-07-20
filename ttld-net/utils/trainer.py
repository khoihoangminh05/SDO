"""End-to-end TTLD-Net training loop (Phase 6)."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data.dataset import create_dataloader
from losses.detection_loss import DetectionLoss
from losses.infonce_loss import InfoNCELoss
from models.ttld_net import TTLDNet
from utils.config import TTLDConfig, resolve_path
from utils.metrics import evaluate_model, save_metrics


def _checkpoint_name(config_path: Path) -> str:
    stem = config_path.stem
    if stem.startswith("m") and "_" in stem:
        return f"{stem}_best.pth"
    return "best_model.pth"


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TTLDConfig) -> LambdaLR:
    """Cosine annealing with linear warmup."""
    warmup = max(cfg.training.warmup_epochs, 0)
    total_epochs = max(cfg.training.epochs, 1)
    eta_min_ratio = 1e-2

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        if total_epochs <= warmup:
            return eta_min_ratio
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _save_checkpoint(
    path: Path,
    model: TTLDNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def train_ttld(
    cfg: TTLDConfig,
    output_dir: Path,
    device: str | int = 0,
    config_path: Path | None = None,
    resume: str | Path | None = None,
    max_epochs: int | None = None,
    max_steps: int | None = None,
) -> Path:
    """
    Train TTLD-Net end-to-end with L_det + L_topology + L_verify.

    Returns path to best checkpoint (.pth).
    """
    from torch.utils.tensorboard import SummaryWriter

    if isinstance(device, str) and device != "cpu":
        device = int(device)
    torch_device = torch.device(f"cuda:{device}" if device != "cpu" else "cpu")

    model = TTLDNet(cfg).to(torch_device)
    if resume:
        state = torch.load(resume, map_location=torch_device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)

    train_yaml = resolve_path(cfg.data.train_yaml)
    val_yaml = resolve_path(cfg.data.val_yaml)

    train_loader = create_dataloader(
        train_yaml,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        pin_memory=cfg.data.pin_memory and torch_device.type == "cuda",
        split="train",
    )
    val_loader = create_dataloader(
        val_yaml,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        pin_memory=cfg.data.pin_memory and torch_device.type == "cuda",
        split="val",
    )

    optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, cfg)

    loss_fns: dict[str, nn.Module] = {
        "detection": DetectionLoss(
            gamma=cfg.loss.focal_gamma,
            alpha=cfg.loss.focal_alpha,
        ).to(torch_device),
        "infonce": InfoNCELoss(temperature=cfg.loss.infonce_temperature).to(torch_device),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    ckpt_name = _checkpoint_name(config_path or Path("m4_full_ttld.yaml"))
    checkpoints_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / ckpt_name
    last_path = output_dir / "last.pth"

    epochs = max_epochs or cfg.training.epochs
    best_recall = -1.0
    global_step = 0

    use_verifier = cfg.model.mode == "full"

    for epoch in range(epochs):
        model.train()
        epoch_losses: dict[str, float] = {}

        for images, targets in train_loader:
            if max_steps is not None and global_step >= max_steps:
                break

            images = images.to(torch_device)
            targets = [t.to(torch_device) for t in targets]

            outputs = model(images)
            losses = model.compute_losses(outputs, targets, loss_fns)

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=cfg.training.grad_clip_norm,
            )
            optimizer.step()

            global_step += 1
            for key, value in losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0.0) + float(value.detach())

            writer.add_scalar("Loss/total", float(losses["total"]), global_step)
            for key in ("det", "topology", "verify"):
                if key in losses:
                    writer.add_scalar(f"Loss/{key}", float(losses[key]), global_step)

            if max_steps is not None and global_step >= max_steps:
                break

        scheduler.step()
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        num_steps = max(len(train_loader), 1)
        if max_steps is not None:
            num_steps = min(num_steps, max_steps)
        for key, total_val in epoch_losses.items():
            writer.add_scalar(f"Epoch/{key}", total_val / num_steps, epoch)

        val_metrics = evaluate_model(
            model,
            val_loader,
            torch_device,
            conf_threshold=cfg.data.eval_conf_threshold,
            use_verifier=use_verifier,
            max_batches=20 if max_steps is not None else None,
        )
        for key, value in val_metrics.items():
            if key in {"ap50", "apsmall", "recall", "precision", "fpr"}:
                writer.add_scalar(f"Metrics/{key}", value, epoch)

        recall = val_metrics.get("recall", 0.0)
        if recall >= best_recall:
            best_recall = recall
            _save_checkpoint(best_path, model, optimizer, epoch, val_metrics)
            save_metrics(val_metrics, output_dir / "best_metrics.json")

        _save_checkpoint(last_path, model, optimizer, epoch, val_metrics)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={epoch_losses.get('total', 0) / num_steps:.4f} | "
            f"recall={val_metrics.get('recall', 0):.4f} | "
            f"ap50={val_metrics.get('ap50', 0):.4f}"
        )

        if max_steps is not None and global_step >= max_steps:
            break

    writer.close()

    if best_path.is_file():
        shutil.copy2(best_path, output_dir / "best.pth")
    return best_path if best_path.is_file() else last_path
