"""Visualization helpers for detections and attention maps."""

from __future__ import annotations

from typing import Any

import numpy as np


def draw_predictions(
    image: np.ndarray,
    boxes: list[list[float]],
    scores: list[float],
    labels: list[int],
    class_names: list[str],
) -> np.ndarray:
    """Draw bounding boxes and class labels on an image."""
    import cv2

    output = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        name = class_names[label] if label < len(class_names) else str(label)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{name}:{score:.2f}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return output


def draw_attention_heatmap(image: np.ndarray, attention_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay an attention heatmap on the source image."""
    import cv2

    heatmap = cv2.applyColorMap(
        (attention_map * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


def draw_deformable_attention(
    image: np.ndarray,
    candidate_center: tuple[int, int],
    offsets: np.ndarray,
    attn_weights: np.ndarray,
) -> np.ndarray:
    """Visualize K deformable sampling points ("tentacles")."""
    import cv2

    output = image.copy()
    cx, cy = candidate_center
    for k, (dx, dy) in enumerate(offsets):
        x, y = int(cx + dx), int(cy + dy)
        weight = float(attn_weights[k])
        color = (int(255 * weight), int(100 * (1 - weight)), 0)
        cv2.circle(output, (x, y), 3, color, -1)
        cv2.line(output, (cx, cy), (x, y), color, 1)
    return output
