from app.models import Detection
from app.utils import calculate_iou, weighted_box_fusion, merge_detections


def test_calculate_iou():
    box1 = (100.0, 100.0, 200.0, 200.0)
    box2 = (100.0, 100.0, 200.0, 225.0)

    iou = calculate_iou(box1, box2)
    # Area1 = 10000, Area2 = 12500, Intersection = 10000
    # Union = 10000 + 12500 - 10000 = 12500
    # IoU = 10000 / 12500 = 0.8
    assert abs(iou - 0.8) < 1e-6


def test_wbf_merges_overlapping_boxes():
    # Two detections of class_id 0 overlapping by exactly 80%
    det1 = Detection(
        class_id=0,
        class_name="Ship",
        confidence=0.9,
        local_bbox=(10.0, 10.0, 110.0, 110.0),
        global_bbox=(100.0, 100.0, 200.0, 200.0),
    )
    det2 = Detection(
        class_id=0,
        class_name="Ship",
        confidence=0.8,
        local_bbox=(10.0, 10.0, 110.0, 135.0),
        global_bbox=(100.0, 100.0, 200.0, 225.0),
    )

    merged = weighted_box_fusion([det1, det2], iou_threshold=0.5)

    # They should be merged into a single box
    assert len(merged) == 1
    fused = merged[0]

    assert fused.class_id == 0
    assert fused.class_name == "Ship"
    assert abs(fused.confidence - 0.85) < 1e-6

    # Merged global bbox calculation:
    # sum_conf = 1.7
    # x_min = (100 * 0.9 + 100 * 0.8) / 1.7 = 100
    # y_min = (100 * 0.9 + 100 * 0.8) / 1.7 = 100
    # x_max = (200 * 0.9 + 200 * 0.8) / 1.7 = 200
    # y_max = (200 * 0.9 + 225 * 0.8) / 1.7 = 360 / 1.7 = 211.76 -> round to 212
    assert fused.global_bbox == (100.0, 100.0, 200.0, 212.0)
    assert fused.local_bbox == (10.0, 10.0, 110.0, 122.0)


def test_wbf_no_merge_different_classes():
    # Two boxes with high overlap but different class IDs should not be merged
    det1 = Detection(
        class_id=0,
        class_name="Ship",
        confidence=0.9,
        local_bbox=(10.0, 10.0, 110.0, 110.0),
        global_bbox=(100.0, 100.0, 200.0, 200.0),
    )
    det2 = Detection(
        class_id=1,
        class_name="Container",
        confidence=0.8,
        local_bbox=(10.0, 10.0, 110.0, 135.0),
        global_bbox=(100.0, 100.0, 200.0, 225.0),
    )

    merged = weighted_box_fusion([det1, det2], iou_threshold=0.5)

    assert len(merged) == 2


def test_merge_detections():
    # Setup overlapping detections
    det1 = Detection(
        class_id=0,
        class_name="Ship",
        confidence=0.9,
        local_bbox=(10.0, 10.0, 110.0, 110.0),
        global_bbox=(100.0, 100.0, 200.0, 200.0),
    )
    det2 = Detection(
        class_id=0,
        class_name="Ship",
        confidence=0.8,
        local_bbox=(10.0, 10.0, 110.0, 135.0),
        global_bbox=(100.0, 100.0, 200.0, 225.0),
    )

    # Image size 1000x1000
    merged = merge_detections(
        [det1, det2],
        image_width=1000.0,
        image_height=1000.0,
        iou_threshold=0.5,
    )

    assert len(merged) == 1
    fused = merged[0]
    assert fused.class_id == 0
    assert fused.class_name == "Ship"
    # WBF averages coordinates. In normalized scale:
    # box1: [0.1, 0.1, 0.2, 0.2]
    # box2: [0.1, 0.1, 0.2, 0.225]
    # fused box y_max: (0.2 * 0.9 + 0.225 * 0.8) / 1.7 = 0.2117647
    # Denormalized y_max: 0.2117647 * 1000 = 211.7647
    assert abs(fused.global_bbox[3] - 211.7647) < 1e-3
    assert abs(fused.confidence - 0.85) < 1e-6
