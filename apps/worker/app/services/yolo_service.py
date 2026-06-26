from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import cv2
from ultralytics import YOLO

import os

from app import config

class YOLOService:
    _instance = None
    model: YOLO = None
    device: str = "cpu"

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(YOLOService, cls).__new__(
                cls, *args, **kwargs
            )
        return cls._instance

    def initialize(self, model_name: str = None) -> None:
        """
        Loads the YOLO model exactly once on the optimal device.
        """
        if self.model is not None:
            return

        if model_name is None:
            model_name = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")

        # Fallback to standard model if custom weights are not found
        if not os.path.exists(model_name) and not model_name.startswith("yolov8n"):
            print(f"Warning: Model {model_name} not found. Falling back to yolov8n.pt")
            model_name = "yolov8n.pt"

        # Optimal device check
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Initialize the YOLO model
        self.model = YOLO(model_name)
        
        # Warmup inference to fuse model and catch any Ultralytics bugs
        try:
            import numpy as np
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(source=dummy_img, conf=0.5, verbose=False)
            print("YOLO model warmup successful.")
        except Exception as e:
            print(f"YOLO model warmup exception (usually safe to ignore): {e}")
        self.model.to(self.device)

    def apply_letterbox(self, image_array: np.ndarray, color=(114, 114, 114)) -> Tuple[np.ndarray, int, int]:
        h, w = image_array.shape[:2]
        max_dim = max(h, w)
        if h == max_dim and w == max_dim:
            return image_array, 0, 0
        
        padded = np.full((max_dim, max_dim, 3), color, dtype=image_array.dtype)
        pad_x = (max_dim - w) // 2
        pad_y = (max_dim - h) // 2
        padded[pad_y:pad_y + h, pad_x:pad_x + w] = image_array
        return padded, pad_x, pad_y

    def run_inference(
        self, image_array: np.ndarray, conf_threshold: float = 0.10, imgsz: int = 1280
    ) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        Runs model.predict on the image array and returns raw predictions:
        (bounding_boxes, confidence_scores, class_ids).
        """
        if self.model is None:
            raise RuntimeError(
                "YOLO model has not been initialized. Call initialize() first."
            )

        padded_image, pad_x, pad_y = self.apply_letterbox(image_array)

        # Run predictions
        results = self.model.predict(
            source=padded_image, conf=conf_threshold, imgsz=imgsz, verbose=False
        )

        boxes: List[List[float]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        if not results:
            return boxes, confidences, class_ids

        # Take the first result
        result = results[0]
        h, w = image_array.shape[:2]

        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                
                # Re-map coordinates and clamp bounds
                lx_min = max(0, xyxy[0] - pad_x)
                ly_min = max(0, xyxy[1] - pad_y)
                lx_max = min(w, xyxy[2] - pad_x)
                ly_max = min(h, xyxy[3] - pad_y)

                # Ignore boxes that fall entirely outside the original crop (in padding)
                if lx_min >= lx_max or ly_min >= ly_max:
                    continue

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                boxes.append([lx_min, ly_min, lx_max, ly_max])
                confidences.append(conf)
                class_ids.append(cls_id)

        return boxes, confidences, class_ids


# Export a single instance
yolo_service = YOLOService()


class SmallObjectDetector:
    """
    Wrapper class for YOLO-P2 small object detection.
    Loads custom trained weights and performs adaptive letterboxing.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.environ.get("YOLO_P2_MODEL_PATH", "runs/detect/train/weights/best.pt")
        
        self.model_path = model_path
        self.model = None
        self.device = "cpu"
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
            
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.model.to(self.device)
                print(f"SmallObjectDetector loaded weights from {model_path}")
            except Exception as e:
                print(f"Error loading SmallObjectDetector weights: {e}")
        else:
            print(f"Warning: SmallObjectDetector weights not found at {model_path}.")

    def adaptive_letterbox(
        self, image: np.ndarray, target_size: int = 640, color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Pads images to square dimensions without distorting the aspect ratio.
        """
        import cv2
        h, w = image.shape[:2]
        
        # Scaling factor
        scale = min(target_size / h, target_size / w)
        nh = int(round(h * scale))
        nw = int(round(w * scale))
        
        # Resize
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # Offsets
        pad_x = (target_size - nw) // 2
        pad_y = (target_size - nh) // 2
        
        # Create padded canvas
        padded = np.full((target_size, target_size, 3), color, dtype=image.dtype)
        padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        
        return padded, scale, (pad_x, pad_y)

    def predict(self, crop: np.ndarray, conf_threshold: float = 0.25) -> List[dict]:
        """
        Runs inference on a crop using the custom YOLO-P2 model.
        """
        if self.model is None:
            print("Error: SmallObjectDetector model is not initialized (weights missing).")
            return []
            
        target_size = 640
        padded_img, scale, (pad_x, pad_y) = self.adaptive_letterbox(crop, target_size=target_size)
        
        # Debug crop dump is gated behind DEBUG_CROPS to avoid disk I/O on every inference.
        if config.DEBUG_CROPS:
            os.makedirs(config.DEBUG_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(config.DEBUG_DIR, "debug_crop_before_yolo.jpg"), padded_img)
        
        # Run inference
        results = self.model.predict(source=padded_img, conf=conf_threshold, verbose=False)
        
        detections = []
        if not results:
            return detections
            
        res = results[0]
        h, w = crop.shape[:2]
        
        if res.boxes is not None:
            for box in res.boxes:
                xyxy = box.xyxy[0].tolist()
                
                # Remap coordinates from padded target_size square back to original crop coordinates
                lx_min = max(0.0, (xyxy[0] - pad_x) / scale)
                ly_min = max(0.0, (xyxy[1] - pad_y) / scale)
                lx_max = min(float(w), (xyxy[2] - pad_x) / scale)
                ly_max = min(float(h), (xyxy[3] - pad_y) / scale)
                
                if lx_min >= lx_max or ly_min >= ly_max:
                    continue
                    
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id] if hasattr(self.model, "names") else str(cls_id)
                
                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [lx_min, ly_min, lx_max, ly_max]
                })
                
        return detections


# ── Phase 5: YOLO26 single-pass inference engine ─────────────────────────────

class YOLO26InferenceEngine:
    """
    Single-pass full-frame inference using the YOLO26-P2 model trained on BSTLD.

    Unlike the recursive crop+upscale pipeline, this engine runs ONE inference
    on the full image at imgsz=1280. YOLO26's built-in STAL+ProgLoss already
    handles tiny objects; the P2 head (stride 4) provides resolution for 8px lights.

    Class mapping (matches bstld.yaml nc=4):
      0 → red    (class_id 91)
      1 → yellow (class_id 92)
      2 → green  (class_id 93)
      3 → off    (class_id 95)

    Activation: set env var PIPELINE_MODE=single_pass (or config.PIPELINE_MODE).
    Falls back gracefully when weights file is missing.
    """

    _BSTLD_CLASS_MAP: Dict[int, Tuple[int, str]] = {
        0: (91, "Red Light"),
        1: (92, "Yellow Light"),
        2: (93, "Green Light"),
        3: (95, "Off Light"),
    }

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[YOLO] = None
        self.device: str = "cpu"
        self.model_path = model_path or os.environ.get(
            "YOLO26_MODEL_PATH", "yolo26p2_color_bstld.pt"
        )
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"

        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self.model.predict(source=dummy, conf=0.5, verbose=False)
                print(f"[YOLO26Engine] Loaded {self.model_path} on {self.device}")
            except Exception as e:
                print(f"[YOLO26Engine] Failed to load model: {e}")
                self.model = None
        else:
            print(
                f"[YOLO26Engine] Weights not found at '{self.model_path}'. "
                "Pipeline will use recursive mode until weights are available."
            )

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict_full_frame(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Run single-pass inference on the full frame.
        Returns list of dicts: {class_id, class_name, confidence, bbox [x1,y1,x2,y2]}
        in original pixel coordinates.
        """
        if not self.ready:
            return []

        conf = conf_threshold if conf_threshold is not None else config.CUSTOM_CONF_THRESHOLD
        img_h, img_w = image.shape[:2]

        results = self.model.predict(
            source=image,
            conf=conf,
            imgsz=config.BASE_IMGSZ,
            verbose=False,
        )

        detections: List[Dict] = []
        if not results:
            return detections

        for box in (results[0].boxes or []):
            xyxy   = box.xyxy[0].tolist()
            conf_v = float(box.conf[0])
            cls_id = int(box.cls[0])

            x1 = max(0.0, min(float(img_w), xyxy[0]))
            y1 = max(0.0, min(float(img_h), xyxy[1]))
            x2 = max(0.0, min(float(img_w), xyxy[2]))
            y2 = max(0.0, min(float(img_h), xyxy[3]))

            if x2 - x1 < config.MIN_BOX_WIDTH or y2 - y1 < config.MIN_BOX_HEIGHT:
                continue

            mapped_id, name = self._BSTLD_CLASS_MAP.get(
                cls_id, (90 + cls_id, f"class_{cls_id}")
            )
            detections.append({
                "class_id":   mapped_id,
                "class_name": name,
                "confidence": conf_v,
                "bbox":       [x1, y1, x2, y2],
            })

        return detections


# Singleton — loaded lazily; only active when PIPELINE_MODE=single_pass
yolo26_engine = YOLO26InferenceEngine()
