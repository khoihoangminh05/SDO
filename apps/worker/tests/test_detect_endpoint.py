import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from app.models import Detection

client = TestClient(app)


@patch("app.main.cv2.imread")
@patch("app.services.recursion_engine.recursion_engine.process_context_aware_recursion")
def test_detect_objects_endpoint(mock_process, mock_imread):
    # Mock image read (1000x1000)
    mock_img = MagicMock()
    mock_img.shape = (1000, 1000, 3)
    mock_img.size = 1000 * 1000 * 3
    mock_imread.return_value = mock_img

    # Recursion engine returns a single traffic-light detection
    mock_process.return_value = [
        Detection(
            class_id=9,
            class_name="traffic light",
            confidence=0.95,
            local_bbox=(10.0, 20.0, 30.0, 40.0),
            global_bbox=(110.0, 220.0, 130.0, 240.0),
            sub_detections=[],
        )
    ]

    payload = {
        "image_id": "img_test_123",
        "image_path": "d:/mock/path/to/image.jpg",
        "zoom_level": 2,
        "global_bbox": [100.0, 200.0, 500.0, 600.0],
    }

    response = client.post("/api/v1/detect", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["image_id"] == "img_test_123"
    assert data["zoom_level"] == 2
    assert data["global_bbox"] == [100.0, 200.0, 500.0, 600.0]
    assert len(data["detections"]) == 1
    det = data["detections"][0]
    assert det["class_id"] == 9
    assert det["class_name"] == "traffic light"
    assert det["confidence"] == 0.95
    assert det["global_bbox"] == [110.0, 220.0, 130.0, 240.0]
    assert data["is_leaf"] is True


def test_detect_objects_invalid_payload():
    # Missing required field image_path
    payload = {
        "image_id": "img_test_123",
        "zoom_level": 2,
        "global_bbox": [100.0, 200.0, 500.0, 600.0],
    }

    response = client.post("/api/v1/detect", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Validation Error)


@patch("app.main.os.path.exists")
@patch("app.main.os.listdir")
@patch("app.main.cv2.imread")
@patch("app.services.recursion_engine.recursion_engine.process_context_aware_recursion")
def test_detect_objects_secure_path_resolution(
    mock_process, mock_imread, mock_listdir, mock_exists
):
    # Resolve the uploaded image by image_id prefix inside /app/uploads
    def exists_side_effect(path_arg):
        return path_arg == "/app/uploads"

    mock_exists.side_effect = exists_side_effect
    mock_listdir.return_value = ["img_test_123.jpg", "other_img.jpg"]

    mock_img = MagicMock()
    mock_img.shape = (1000, 1000, 3)
    mock_img.size = 1000 * 1000 * 3
    mock_imread.return_value = mock_img

    mock_process.return_value = []

    payload = {
        "image_id": "img_test_123",
        "image_path": "d:/mock/path/to/image.jpg",
        "zoom_level": 2,
        "global_bbox": [100.0, 200.0, 500.0, 600.0],
    }

    response = client.post("/api/v1/detect", json=payload)
    assert response.status_code == 200

    # cv2.imread must be called with the securely resolved path, not the payload path
    mock_imread.assert_called_once_with(
        os.path.normpath("/app/uploads/img_test_123.jpg")
    )
