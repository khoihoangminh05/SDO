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
from utils.fast_train import apply_fast_profile, apply_proplus_profile
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
    parser.add_argument("--max-dets", type=int, default=100, help="Max detections per image")
    parser.add_argument("--fast", action="store_true", help="Match FAST train image size/settings")
    parser.add_argument("--proplus", action="store_true", help="Match PRO+ train image size/settings")
    return parser.parse_args()


def _load_ttld_model(cfg, weights: Path, device: torch.device) -> TTLDNet:
    model = TTLDNet(cfg)
    # Materialize lazy backbone/neck so checkpoint keys match.
    model.materialize(device, tuple(cfg.data.image_size))
    state = torch.load(weights, map_location=device, weights_only=False)
    payload = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(payload, strict=False)
    print(
        f"Loaded {weights.name} | missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if len(missing) > 50:
        print(
            "WARNING: many missing keys — checkpoint may predate materialize fix. "
            "Retrain required.",
            flush=True,
        )
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.proplus and args.fast:
        raise SystemExit("Use only one of --fast or --proplus")
    if args.proplus:
        apply_proplus_profile(cfg)
    elif args.fast:
        apply_fast_profile(cfg)

    weights = resolve_path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    device: str | int = int(args.device) if args.device.isdigit() else args.device
    torch_device = torch.device(f"cuda:{device}" if device != "cpu" else "cpu")
    if args.conf is not None:
        eval_conf = args.conf
    elif cfg.model.mode == "baseline":
        eval_conf = cfg.data.eval_conf_threshold
    else:
        # For TTLD variants (M1-M4), high-recall behavior is evaluated at low conf.
        eval_conf = min(cfg.data.conf_threshold, cfg.data.eval_conf_threshold)

    if cfg.model.mode == "baseline":
        metrics = evaluate_baseline(weights, cfg, conf=eval_conf, device=device)
    elif cfg.model.mode in {"shallow", "shallow_focal", "topology", "full"}:
        model = _load_ttld_model(cfg, weights, torch_device)
        val_yaml = resolve_path(cfg.data.val_yaml)
        val_batch = getattr(cfg.training, "val_batch_size", None) or min(cfg.training.batch_size, 2)
        image_size = tuple(cfg.data.image_size)
        val_loader = create_dataloader(
            val_yaml,
            batch_size=val_batch,
            num_workers=cfg.data.num_workers,
            shuffle=False,
            pin_memory=False,
            split="val",
            image_size=image_size,
        )
        metrics = evaluate_model(
            model,
            val_loader,
            torch_device,
            conf_threshold=eval_conf,
            use_verifier=cfg.model.mode == "full",
            max_batches=args.max_batches,
            max_dets=args.max_dets,
        )
        metrics["eval_conf"] = float(eval_conf)
        metrics["mode"] = cfg.model.mode
        metrics["image_size"] = list(image_size)
        metrics["max_dets"] = int(args.max_dets)
    else:
        raise NotImplementedError(f"Evaluation mode '{cfg.model.mode}' not implemented.")

    save_metrics(metrics, args.output)
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
