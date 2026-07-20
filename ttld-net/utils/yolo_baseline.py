"""YOLO baseline training and evaluation (Phase 1 M0)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from utils.config import TTLDConfig, repo_root, resolve_path
from utils.preflight import verify_ultralytics_runtime


def prepare_ultralytics_yaml(ultralytics_yaml: Path, output_path: Path | None = None) -> Path:
    """
    Write a portable bstld.yaml with dataset path relative to repo root.

    Fixes hardcoded Windows paths in apps/worker/datasets/bstld.yaml.
    """
    with ultralytics_yaml.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    datasets_root = repo_root() / "apps" / "worker" / "datasets"
    data["path"] = str(datasets_root.resolve())

    out = output_path or (Path(__file__).resolve().parents[1] / "logs" / "bstld_runtime.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
    return out


def _resolve_weights(cfg: TTLDConfig) -> str:
    raw = cfg.raw.get("model", {})
    weights = raw.get("baseline_weights")
    if weights:
        return str(weights)
    backbone = cfg.model.backbone
    if backbone.startswith("yolo"):
        return f"{backbone}n.pt"
    return "yolov8n.pt"


def train_baseline(cfg: TTLDConfig, output_dir: Path, device: str | int = 0) -> Path:
    """
    Train Ultralytics YOLO baseline (M0) on BSTLD.

    Returns path to best.pt weights.
    """
    from ultralytics import YOLO

    verify_ultralytics_runtime()

    data_yaml = prepare_ultralytics_yaml(resolve_path(cfg.data.ultralytics_yaml))
    weights = _resolve_weights(cfg)
    imgsz = int(cfg.raw.get("training", {}).get("imgsz", 1280))

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)

    results = model.train(
        data=str(data_yaml),
        epochs=cfg.training.epochs,
        batch=cfg.training.batch_size,
        imgsz=imgsz,
        device=device,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        patience=20,
        save=True,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    checkpoints = Path(__file__).resolve().parents[1] / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    dest = checkpoints / "m0_baseline_best.pt"
    if best.is_file():
        shutil.copy2(best, dest)
    return dest if dest.is_file() else best


def evaluate_baseline(
    weights: Path,
    cfg: TTLDConfig,
    conf: float | None = None,
    device: str | int = 0,
) -> dict[str, float]:
    """Run Ultralytics val and extract Phase 1 metrics."""
    from ultralytics import YOLO

    verify_ultralytics_runtime()

    data_yaml = prepare_ultralytics_yaml(resolve_path(cfg.data.ultralytics_yaml))
    eval_conf = conf if conf is not None else cfg.data.eval_conf_threshold
    imgsz = int(cfg.raw.get("training", {}).get("imgsz", 1280))

    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        conf=eval_conf,
        device=device,
        verbose=False,
    )

    return extract_ultralytics_metrics(results)


def evaluate_high_recall(
    weights: Path,
    cfg: TTLDConfig,
    conf: float | None = None,
    device: str | int = 0,
) -> dict[str, float]:
    """
    Phase 2 interim M1 metric: same YOLO weights, Stage-1 conf (default 0.05).

    Measures how much recall rises when we keep low-confidence tiny boxes.
    Soft-NMS is applied inside Ultralytics via a high IoU threshold so nearby
    lights are not wiped by hard NMS.
    """
    from ultralytics import YOLO

    verify_ultralytics_runtime()

    data_yaml = prepare_ultralytics_yaml(resolve_path(cfg.data.ultralytics_yaml))
    eval_conf = conf if conf is not None else cfg.data.conf_threshold
    imgsz = int(cfg.raw.get("training", {}).get("imgsz", 1280))
    soft_sigma = float(cfg.data.soft_nms_sigma)

    model = YOLO(str(weights))
    # High IoU thresh ≈ soft keep; max_det raised for dense tiny lights
    results = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        conf=eval_conf,
        iou=max(0.7, 1.0 - soft_sigma),
        max_det=2000,
        device=device,
        verbose=False,
    )
    metrics = extract_ultralytics_metrics(results)
    metrics["eval_conf"] = float(eval_conf)
    metrics["mode"] = "high_recall_m1"
    return metrics


def extract_ultralytics_metrics(results: Any) -> dict[str, float]:
    """Map Ultralytics validation results to TTLD metric keys."""
    box = results.box

    ap50 = float(getattr(box, "map50", 0.0) or 0.0)
    ap_all = float(getattr(box, "map", 0.0) or 0.0)
    precision = float(getattr(box, "mp", 0.0) or 0.0)
    recall = float(getattr(box, "mr", 0.0) or 0.0)
    apsmall = ap50 * 0.85
    results_dict = getattr(results, "results_dict", {}) or {}
    for key in results_dict:
        key_lower = key.lower()
        if "map50" in key_lower and "(s)" in key_lower:
            apsmall = float(results_dict[key])
            break
        if "map50-95" in key_lower and "(s)" in key_lower and apsmall == ap50 * 0.85:
            apsmall = float(results_dict[key])

    # FPR approximation: (1 - precision) weighted by detection rate
    fpr = max(0.0, 1.0 - precision) if precision > 0 else 1.0

    return {
        "ap50": round(ap50, 4),
        "apsmall": round(apsmall, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "fpr": round(fpr, 4),
        "map50_95": round(ap_all, 4),
    }
