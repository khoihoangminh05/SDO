"""Evaluation metrics for TTLD-Net."""

from __future__ import annotations

from typing import Any


def compute_metrics(preds: Any, targets: Any, conf_threshold: float = 0.5) -> dict[str, float]:
    """
    Compute AP50, APsmall, recall, precision, FPR.

    Phase 1/6 will implement COCO-style evaluation on Bosch holdout.
    """
    raise NotImplementedError("Phase 1 (T1.6) / Phase 6: implement metric computation.")


def save_metrics(metrics: dict[str, float], output_path: str) -> None:
    """Persist metrics JSON for ablation comparison."""
    import json
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
