from typing import List, Tuple
from app.models import Detection

BoundingBox = Tuple[float, float, float, float]


def local_to_global_bbox(
    local_bbox: BoundingBox, tile_bbox: BoundingBox
) -> List[int]:
    """
    Converts local coordinates (relative to a cropped tile) back to global coordinates.
    Rounds float coordinate values to the nearest integer.
    """
    lx_min = round(float(local_bbox[0]))
    ly_min = round(float(local_bbox[1]))
    lx_max = round(float(local_bbox[2]))
    ly_max = round(float(local_bbox[3]))

    tx_min = round(float(tile_bbox[0]))
    ty_min = round(float(tile_bbox[1]))

    gx_min = tx_min + lx_min
    gy_min = ty_min + ly_min
    gx_max = tx_min + lx_max
    gy_max = ty_min + ly_max

    return [int(gx_min), int(gy_min), int(gx_max), int(gy_max)]


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Calculates Intersection over Union (IoU) of two bounding boxes.
    """
    x1_min, y1_min, x1_max, y1_max = [float(v) for v in box1]
    x2_min, y2_min, x2_max, y2_max = [float(v) for v in box2]

    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def weighted_box_fusion(
    detections: List[Detection], iou_threshold: float = 0.55
) -> List[Detection]:
    """
    Groups overlapping bounding boxes for each class using IoU and fuses them
    weighted by their confidence scores.
    """
    if not detections:
        return []

    # Group detections by class_id
    grouped = {}
    for d in detections:
        grouped.setdefault(d.class_id, []).append(d)

    final_detections = []

    for class_id, class_dets in grouped.items():
        # Sort by confidence descending
        class_dets = sorted(
            class_dets, key=lambda x: x.confidence, reverse=True
        )

        clusters = []

        for det in class_dets:
            matched = False
            for cluster in clusters:
                # Calculate the weighted average box of the cluster
                sum_conf = sum(c.confidence for c in cluster)
                avg_box = (
                    sum(c.global_bbox[0] * c.confidence for c in cluster)
                    / sum_conf,
                    sum(c.global_bbox[1] * c.confidence for c in cluster)
                    / sum_conf,
                    sum(c.global_bbox[2] * c.confidence for c in cluster)
                    / sum_conf,
                    sum(c.global_bbox[3] * c.confidence for c in cluster)
                    / sum_conf,
                )

                if calculate_iou(det.global_bbox, avg_box) >= iou_threshold:
                    cluster.append(det)
                    matched = True
                    break

            if not matched:
                clusters.append([det])

        # Merge boxes in each cluster
        for cluster in clusters:
            sum_conf = sum(c.confidence for c in cluster)
            merged_global = (
                float(
                    round(
                        sum(c.global_bbox[0] * c.confidence for c in cluster)
                        / sum_conf
                    )
                ),
                float(
                    round(
                        sum(c.global_bbox[1] * c.confidence for c in cluster)
                        / sum_conf
                    )
                ),
                float(
                    round(
                        sum(c.global_bbox[2] * c.confidence for c in cluster)
                        / sum_conf
                    )
                ),
                float(
                    round(
                        sum(c.global_bbox[3] * c.confidence for c in cluster)
                        / sum_conf
                    )
                ),
            )

            merged_local = (
                float(
                    round(
                        sum(c.local_bbox[0] * c.confidence for c in cluster)
                        / sum_conf
                    )
                ),
                float(
                    round(
                        sum(c.local_bbox[1] * c.confidence for c in cluster)
                        / sum_conf
                    )
                ),
                float(
                    round(
                        sum(c.local_bbox[2] * c.confidence for c in cluster)
                        / sum_conf
                    )
                ),
                float(
                    round(
                        sum(c.local_bbox[3] * c.confidence for c in cluster)
                        / sum_conf
                    )
                ),
            )

            merged_conf = sum(c.confidence for c in cluster) / len(cluster)

            first = cluster[0]
            final_detections.append(
                Detection(
                    class_id=first.class_id,
                    class_name=first.class_name,
                    confidence=float(merged_conf),
                    local_bbox=merged_local,
                    global_bbox=merged_global,
                )
            )

    return final_detections


def merge_detections(
    detections_list: List[Detection],
    image_width: float,
    image_height: float,
    iou_threshold: float = 0.3,
) -> List[Detection]:
    """
    Fuses overlapping bounding boxes across leaf nodes using Weighted Box Fusion
    from the ensemble_boxes library.
    """
    if not detections_list:
        return []

    from ensemble_boxes import weighted_boxes_fusion

    normalized_boxes = []
    scores = []
    labels = []

    for det in detections_list:
        # Normalize coordinates to [0, 1] for ensemble_boxes, clamping just in case
        x_min = max(0.0, min(1.0, float(det.global_bbox[0]) / image_width))
        y_min = max(0.0, min(1.0, float(det.global_bbox[1]) / image_height))
        x_max = max(0.0, min(1.0, float(det.global_bbox[2]) / image_width))
        y_max = max(0.0, min(1.0, float(det.global_bbox[3]) / image_height))

        if x_min >= x_max or y_min >= y_max:
            continue

        normalized_boxes.append([x_min, y_min, x_max, y_max])
        scores.append(float(det.confidence))
        labels.append(int(det.class_id))

    if not normalized_boxes:
        return []

    # WBF expects lists of lists (one per model)
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        [normalized_boxes],
        [scores],
        [labels],
        weights=None,
        iou_thr=iou_threshold,
        skip_box_thr=0.0
    )

    class_name_map = {d.class_id: d.class_name for d in detections_list}
    first_det = detections_list[0]
    tile_x_min = first_det.global_bbox[0] - first_det.local_bbox[0]
    tile_y_min = first_det.global_bbox[1] - first_det.local_bbox[1]

    fused_detections = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        gx_min = float(box[0] * image_width)
        gy_min = float(box[1] * image_height)
        gx_max = float(box[2] * image_width)
        gy_max = float(box[3] * image_height)

        lx_min = gx_min - tile_x_min
        ly_min = gy_min - tile_y_min
        lx_max = gx_max - tile_x_min
        ly_max = gy_max - tile_y_min

        class_id = int(label)
        class_name = class_name_map.get(class_id, f"class_{class_id}")

        fused_detections.append(
            Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(score),
                local_bbox=(lx_min, ly_min, lx_max, ly_max),
                global_bbox=(gx_min, gy_min, gx_max, gy_max),
            )
        )

    return fused_detections
