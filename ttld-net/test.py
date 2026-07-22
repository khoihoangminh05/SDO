#!/usr/bin/env python3
"""TTLD-Net evaluation entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import create_dataloader
from models.ttld_net import TTLDNet
from utils.config import load_config, resolve_path
from utils.metrics import evaluate_model, save_metrics
from utils.yolo_baseline import evaluate_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TTLD-Net")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--output", required=True, help="Metrics JSON output path")
    parser.add_argument("--conf", type=float, default=None, help="Override eval confidence threshold")
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit val batches (smoke test)")
    return parser.parse_args()


def _load_ttld_model(cfg, weights: Path, device: torch.device) -> TTLDNet:
    model = TTLDNet(cfg)
    state = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    weights = resolve_path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    device: str | int = int(args.device) if args.device.isdigit() else args.device
    torch_device = torch.device(f"cuda:{device}" if device != "cpu" else "cpu")
    eval_conf = args.conf if args.conf is not None else cfg.data.eval_conf_threshold

    if cfg.model.mode == "baseline":
        metrics = evaluate_baseline(weights, cfg, conf=eval_conf, device=device)
    elif cfg.model.mode in {"shallow", "shallow_focal", "topology", "full"}:
        model = _load_ttld_model(cfg, weights, torch_device)
        val_yaml = resolve_path(cfg.data.val_yaml)
        val_batch = getattr(cfg.training, "val_batch_size", None) or min(cfg.training.batch_size, 2)
        val_loader = create_dataloader(
            val_yaml,
            batch_size=val_batch,
            num_workers=cfg.data.num_workers,
            shuffle=False,
            pin_memory=False,
            split="val",
        )
        metrics = evaluate_model(
            model,
            val_loader,
            torch_device,
            conf_threshold=eval_conf,
            use_verifier=cfg.model.mode == "full",
            max_batches=args.max_batches,
        )
        metrics["eval_conf"] = float(eval_conf)
        metrics["mode"] = cfg.model.mode
    else:
        raise NotImplementedError(f"Evaluation mode '{cfg.model.mode}' not implemented.")

    save_metrics(metrics, args.output)
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
