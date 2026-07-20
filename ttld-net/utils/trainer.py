"""End-to-end TTLD-Net training loop (Phase 6)."""

from __future__ import annotations

import gc
import math
import shutil
from contextlib import nullcontext
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


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _loss_has_nan(losses: dict[str, torch.Tensor]) -> bool:
    for value in losses.values():
        if not torch.isfinite(value).all():
            return True
    return False


def _format_losses(losses: dict[str, torch.Tensor]) -> str:
    parts = []
    for key in ("total", "det", "topology", "verify"):
        if key in losses:
            parts.append(f"{key}={float(losses[key].detach()):.4f}")
    return " | ".join(parts)


def _grads_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return False
    return True


def _outputs_finite(outputs: dict[str, Any]) -> bool:
    for key in ("fcand", "zi", "valid_logits"):
        tensor = outputs.get(key)
        if tensor is not None and not torch.isfinite(tensor).all():
            return False
    for scale in outputs.get("raw_outputs") or []:
        for part in ("obj", "cls", "box"):
            tensor = scale.get(part)
            if tensor is not None and not torch.isfinite(tensor).all():
                return False
    return True


def _diagnose_outputs(outputs: dict[str, Any]) -> str:
    bad: list[str] = []
    for key in ("fcand", "zi", "valid_logits"):
        tensor = outputs.get(key)
        if tensor is not None and not torch.isfinite(tensor).all():
            bad.append(key)
    for idx, scale in enumerate(outputs.get("raw_outputs") or []):
        for part in ("obj", "cls", "box"):
            tensor = scale.get(part)
            if tensor is not None and not torch.isfinite(tensor).all():
                bad.append(f"raw[{idx}].{part}")
    return ", ".join(bad) if bad else "unknown"


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

    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    model = TTLDNet(cfg).to(torch_device)
    if resume:
        state = torch.load(resume, map_location=torch_device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)

    train_yaml = resolve_path(cfg.data.train_yaml)
    val_yaml = resolve_path(cfg.data.val_yaml)
    val_batch_size = getattr(cfg.training, "val_batch_size", None) or min(cfg.training.batch_size, 2)

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
        batch_size=val_batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        pin_memory=cfg.data.pin_memory and torch_device.type == "cuda",
        split="val",
    )

    optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, cfg)
    use_amp = bool(getattr(cfg.training, "use_amp", False)) and torch_device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    aux_warmup = int(getattr(cfg.training, "aux_loss_warmup_steps", 5))

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
    consecutive_bad = 0

    use_verifier = cfg.model.mode == "full"

    print(
        f"Training on {torch_device} | batch_size={cfg.training.batch_size} | "
        f"val_batch_size={val_batch_size} | max_candidates={model.generator.max_candidates} | "
        f"amp={use_amp} | lr={cfg.training.lr} | aux_warmup={aux_warmup}"
    )

    # Sanity-check one batch before training.
    model.train()
    for sanity_images, _ in train_loader:
        sanity_images = sanity_images.to(torch_device, non_blocking=True)
        with _autocast_context(torch_device, use_amp):
            sanity_out = model(sanity_images)
        if not _outputs_finite(sanity_out):
            raise RuntimeError(
                f"Forward pass produces non-finite values before training: "
                f"{_diagnose_outputs(sanity_out)}. Try --no-amp or lower --batch-size."
            )
        break

    for epoch in range(epochs):
        model.train()
        epoch_losses: dict[str, float] = {}
        steps_this_epoch = 0

        for images, targets in train_loader:
            if max_steps is not None and global_step >= max_steps:
                break

            images = images.to(torch_device, non_blocking=True)
            targets = [t.to(torch_device) for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(torch_device, use_amp):
                outputs = model(images)

            if not _outputs_finite(outputs):
                consecutive_bad += 1
                print(
                    f"WARNING: non-finite forward at step {global_step + 1}: "
                    f"{_diagnose_outputs(outputs)} (skipped {consecutive_bad}x)"
                )
                optimizer.zero_grad(set_to_none=True)
                del outputs
                if consecutive_bad >= 20:
                    raise RuntimeError(
                        "Training aborted: forward outputs became non-finite. "
                        "Pull latest code (use_amp=false) and retry with --no-amp."
                    )
                continue

            lambda1 = 0.0 if global_step < aux_warmup else cfg.loss.lambda1
            lambda2 = 0.0 if global_step < aux_warmup else cfg.loss.lambda2

            # Losses in fp32 — avoids NaN from fp16 BCE/InfoNCE on large tensors.
            with _autocast_context(torch_device, enabled=False):
                losses = model.compute_losses(
                    outputs,
                    targets,
                    loss_fns,
                    lambda1=lambda1,
                    lambda2=lambda2,
                )

            if _loss_has_nan(losses):
                consecutive_bad += 1
                print(
                    f"WARNING: non-finite loss at step {global_step + 1}: "
                    f"{_format_losses(losses)} (skipped {consecutive_bad}x)"
                )
                optimizer.zero_grad(set_to_none=True)
                del outputs, losses
                if consecutive_bad >= 20:
                    raise RuntimeError(
                        "Training aborted: 20 consecutive batches with non-finite loss. "
                        "Try --batch-size 2 and check dataset labels."
                    )
                continue

            if not losses["total"].requires_grad:
                consecutive_bad += 1
                print(
                    f"WARNING: loss has no grad at step {global_step + 1}: "
                    f"{_format_losses(losses)} (skipped {consecutive_bad}x)"
                )
                optimizer.zero_grad(set_to_none=True)
                del outputs, losses
                continue

            consecutive_bad = 0
            if use_amp:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
            else:
                losses["total"].backward()

            if not _grads_finite(model):
                print(
                    f"WARNING: non-finite gradients at step {global_step + 1}, "
                    f"skipping optimizer step ({_format_losses(losses)})"
                )
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.update()
                del outputs, losses
                continue

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=cfg.training.grad_clip_norm,
            )
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            global_step += 1
            steps_this_epoch += 1
            for key, value in losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0.0) + float(value.detach())

            writer.add_scalar("Loss/total", float(losses["total"]), global_step)
            for key in ("det", "topology", "verify"):
                if key in losses:
                    writer.add_scalar(f"Loss/{key}", float(losses[key]), global_step)

            del outputs, losses

            if max_steps is not None and global_step >= max_steps:
                break

        scheduler.step()
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        num_steps = max(steps_this_epoch, 1)
        for key, total_val in epoch_losses.items():
            writer.add_scalar(f"Epoch/{key}", total_val / num_steps, epoch)

        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

        val_metrics = evaluate_model(
            model,
            val_loader,
            torch_device,
            conf_threshold=cfg.data.conf_threshold if max_steps is not None else cfg.data.eval_conf_threshold,
            use_verifier=use_verifier if max_steps is None else False,
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
