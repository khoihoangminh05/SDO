import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.yolo_service import YOLOService, yolo_service


def test_yolo_service_singleton():
    service1 = YOLOService()
    service2 = YOLOService()
    assert service1 is service2
    assert service1 is yolo_service


@patch("app.services.yolo_service.YOLO")
@patch("app.services.yolo_service.torch")
def test_yolo_service_initialize(mock_torch, mock_yolo):
    # Setup mock torch
    mock_torch.cuda.is_available.return_value = True

    # Reset model to force re-initialization
    service = YOLOService()
    service.model = None

    service.initialize(model_name="yolov8n.pt")

    assert service.model is not None
    assert service.device == "cuda"
    mock_yolo.assert_called_once_with("yolov8n.pt")
    service.model.to.assert_called_once_with("cuda")


@patch("app.services.yolo_service.YOLO")
@patch("app.services.yolo_service.torch")
def test_yolo_service_initialize_mps(mock_torch, mock_yolo):
    # Setup mock torch for mps
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True

    service = YOLOService()
    service.model = None

    service.initialize(model_name="yolov8n.pt")

    assert service.device == "mps"


@patch("app.services.yolo_service.YOLO")
@patch("app.services.yolo_service.torch")
def test_yolo_service_initialize_cpu(mock_torch, mock_yolo):
    # Setup mock torch for cpu fallback
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False

    service = YOLOService()
    service.model = None

    service.initialize(model_name="yolov8n.pt")

    assert service.device == "cpu"


def test_run_inference_not_initialized():
    service = YOLOService()
    # Mock model to be None
    with patch.object(service, "model", None):
        with pytest.raises(RuntimeError) as exc_info:
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            service.run_inference(dummy_img)
        assert "YOLO model has not been initialized" in str(exc_info.value)


def test_run_inference_success():
    service = YOLOService()

    # Mock model.predict return value
    mock_result = MagicMock()

    # Mock boxes
    mock_box1 = MagicMock()
    mock_box1.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 30.0, 40.0])]
    mock_box1.conf = [0.9]
    mock_box1.cls = [2]

    mock_box2 = MagicMock()
    mock_box2.xyxy = [MagicMock(tolist=lambda: [50.0, 60.0, 70.0, 80.0])]
    mock_box2.conf = [0.8]
    mock_box2.cls = [5]

    mock_result.boxes = [mock_box1, mock_box2]

    mock_predict = MagicMock(return_value=[mock_result])

    with patch.object(service, "model") as mock_model:
        mock_model.predict = mock_predict

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes, confidences, class_ids = service.run_inference(
            dummy_img, conf_threshold=0.5, imgsz=1280
        )

        # A 100x100 square is already square, so letterbox returns it unchanged.
        mock_predict.assert_called_once_with(
            source=dummy_img, conf=0.5, imgsz=1280, verbose=False
        )
        assert boxes == [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]
        assert confidences == [0.9, 0.8]
        assert class_ids == [2, 5]
