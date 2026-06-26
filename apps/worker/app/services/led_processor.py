import cv2
import numpy as np
import easyocr
import torch
from typing import List
from app.models import Detection

# Global EasyOCR Reader instance (Singleton pattern)
_ocr_reader_instance = None

def get_ocr_reader() -> easyocr.Reader:
    """
    Initializes and returns the easyocr.Reader instance.
    Utilizes a global singleton to prevent reloading the model weights on subsequent calls.
    """
    global _ocr_reader_instance
    if _ocr_reader_instance is None:
        gpu_available = torch.cuda.is_available()
        _ocr_reader_instance = easyocr.Reader(['en'], gpu=gpu_available)
    return _ocr_reader_instance


def restore_and_read_led(
    traffic_light_crop: np.ndarray,
    parent_global_bbox: List[float]
) -> List[Detection]:
    """
    Processes a traffic light crop to binarize and close gaps in 7-segment LED numbers,
    performs OCR using EasyOCR, and maps detections back to global coordinates.
    
    Args:
        traffic_light_crop (np.ndarray): Image crop of the traffic light box.
        parent_global_bbox (List[float]): Parent bounding box coordinates [x_min, y_min, x_max, y_max] globally.
        
    Returns:
        List[Detection]: List of formatted Detection objects containing the detected LED digits.
    """
    if traffic_light_crop is None or traffic_light_crop.size == 0:
        return []

    # 1. Grayscale conversion
    gray = cv2.cvtColor(traffic_light_crop, cv2.COLOR_BGR2GRAY)

    # 2. Binarization (Otsu's thresholding) to isolate the bright LED lights
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Morphological Closing to bridge black gaps in 7-segment LED digits
    # Using a 5x5 elliptical structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 4. OCR Reader inference with digit-only whitelist
    reader = get_ocr_reader()
    ocr_results = reader.readtext(closed, allowlist='0123456789')

    detections = []
    parent_x_min, parent_y_min, parent_x_max, parent_y_max = parent_global_bbox

    for res in ocr_results:
        pts, text, conf = res
        text_clean = "".join([c for c in text if c.isdigit()])
        if not text_clean:
            continue

        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        
        lx_min = float(min(xs))
        ly_min = float(min(ys))
        lx_max = float(max(xs))
        ly_max = float(max(ys))

        # Map local coordinates relative to the crop back to global coordinates
        gx_min = parent_x_min + lx_min
        gy_min = parent_y_min + ly_min
        gx_max = parent_x_min + lx_max
        gy_max = parent_y_min + ly_max

        detections.append(
            Detection(
                class_id=94,
                class_name=f"digit_{text_clean}",
                confidence=float(conf),
                local_bbox=(lx_min, ly_min, lx_max, ly_max),
                global_bbox=(gx_min, gy_min, gx_max, gy_max)
            )
        )

    return detections
