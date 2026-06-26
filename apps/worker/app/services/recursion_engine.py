import cv2
import numpy as np
import logging
import os
from typing import List, Tuple, Optional
from app import config
from app.models import Detection
from app.services.yolo_service import yolo_service, SmallObjectDetector, yolo26_engine
from app.services.ocr_service import ocr_service
from app.utils import weighted_box_fusion

logger = logging.getLogger("app")

class SpatialGraphNode:
    """
    Represents a node in the Context-Aware Spatial Graph.
    Hierarchy: Intersection -> Traffic Light Pole -> Lamp Box -> LED Digits
    """
    def __init__(self, name: str, bbox: List[float], confidence: float, class_id: int):
        self.name = name
        self.bbox = bbox  # [x_min, y_min, x_max, y_max] (global coordinates)
        self.confidence = confidence
        self.class_id = class_id
        self.children: List['SpatialGraphNode'] = []

    def add_child(self, child: 'SpatialGraphNode'):
        self.children.append(child)


class FlopsTracker:
    """
    Utility class to count actual YOLO inferences versus standard 4x4 grid split inferences (21).
    """
    def __init__(self):
        self.actual_inferences = 0
        self.grid_inferences = 21

    def increment(self, count: int = 1):
        self.actual_inferences += count

    def get_compute_saved(self) -> float:
        if self.grid_inferences <= 0:
            return 0.0
        saved = (1.0 - (self.actual_inferences / self.grid_inferences)) * 100.0
        return float(saved)

    def log_metrics(self):
        saved_pct = self.get_compute_saved()
        logger.info(
            f"[FlopsTracker] Actual Inferences: {self.actual_inferences}, "
            f"Grid Inferences: {self.grid_inferences}, "
            f"Compute Saved: {saved_pct:.2f}%"
        )


class RecursionEngine:
    def __init__(self):
        self.custom_detector = None
        self.graph = {
            'intersection': ['traffic_light_pole'],
            'traffic_light_pole': ['traffic_light_box'],
            'traffic_light_box': ['led_digit']
        }

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the luminance (L) channel in LAB space. This boosts local
        contrast for low-light / back-lit scenes while leaving hue untouched, so
        the downstream HSV colour thresholds stay valid.
        """
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=config.CLAHE_CLIP_LIMIT,
                tileGridSize=(config.CLAHE_TILE_GRID, config.CLAHE_TILE_GRID),
            )
            l = clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.warning(f"[CLAHE] Skipped enhancement due to error: {e}")
            return image

    def _preprocess_for_ocr(self, box_crop: np.ndarray) -> np.ndarray:
        """
        Convert a colour-light crop into a binary image suitable for digit OCR.
        Steps:
          1. Isolate coloured pixels (red/yellow/green) via HSV masking.
          2. Close small gaps (morphological close).
          3. Upscale for EasyOCR.
          4. Threshold + pad.
        Returns a padded binary image ready for EasyOCR.
        """
        hsv = cv2.cvtColor(box_crop, cv2.COLOR_BGR2HSV)
        mask_r1 = cv2.inRange(hsv, np.array(config.OCR_HSV_RED1_LOWER),
                                    np.array(config.OCR_HSV_RED1_UPPER))
        mask_r2 = cv2.inRange(hsv, np.array(config.OCR_HSV_RED2_LOWER),
                                    np.array(config.OCR_HSV_RED2_UPPER))
        mask_y  = cv2.inRange(hsv, np.array(config.OCR_HSV_YELLOW_LOWER),
                                    np.array(config.OCR_HSV_YELLOW_UPPER))
        mask_g  = cv2.inRange(hsv, np.array(config.OCR_HSV_GREEN_LOWER),
                                    np.array(config.OCR_HSV_GREEN_UPPER))
        color_mask = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2),
                                    cv2.bitwise_or(mask_y, mask_g))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        upscaled = cv2.resize(closed, (0, 0),
                              fx=config.OCR_UPSCALE_FACTOR,
                              fy=config.OCR_UPSCALE_FACTOR,
                              interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(upscaled, 127, 255, cv2.THRESH_BINARY)
        return cv2.copyMakeBorder(thresh, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)

    def _validate_digit(self, text: str) -> Optional[str]:
        """Return digit string if it's a valid countdown number, else None."""
        digits = "".join(c for c in text if c.isdigit())
        if not digits:
            return None
        try:
            val = int(digits)
        except ValueError:
            return None
        if config.OCR_DIGIT_MIN <= val <= config.OCR_DIGIT_MAX:
            return digits
        logger.debug(f"[OCR] Digit {val} out of valid range [{config.OCR_DIGIT_MIN},{config.OCR_DIGIT_MAX}]")
        return None

    def _run_ocr_on_box(
        self,
        box_det,
        pole_det,
        upscaled: np.ndarray,
        scale: float,
        pgx_min: float,
        pgy_min: float,
        pw: float,
        ph: float,
    ) -> None:
        """
        Run EasyOCR digit detection on a single colour-light box crop.
        Appends digit Detection objects to box_det.sub_detections and
        pole_det.sub_detections (flat list).
        """
        if ocr_service.reader is None:
            return

        blx_min, bly_min, blx_max, bly_max = box_det.local_bbox
        b_xmin = max(0, min(upscaled.shape[1], int(round(blx_min * scale))))
        b_ymin = max(0, min(upscaled.shape[0], int(round(bly_min * scale))))
        b_xmax = max(0, min(upscaled.shape[1], int(round(blx_max * scale))))
        b_ymax = max(0, min(upscaled.shape[0], int(round(bly_max * scale))))

        if b_xmin >= b_xmax or b_ymin >= b_ymax:
            return
        box_crop_img = upscaled[b_ymin:b_ymax, b_xmin:b_xmax]
        if box_crop_img.size == 0:
            return

        try:
            padded = self._preprocess_for_ocr(box_crop_img)
            ocr_results = ocr_service.reader.readtext(padded, allowlist="0123456789")

            for pts, text, conf in ocr_results:
                if conf < config.OCR_CONF_THRESHOLD:
                    continue
                clean = self._validate_digit(text)
                if clean is None:
                    continue

                pad = 30
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                # Undo padding + OCR upscale → local box coords
                ocr_s = config.OCR_UPSCALE_FACTOR
                bx_min_l = max(0.0, (min(xs) - pad) / ocr_s)
                by_min_l = max(0.0, (min(ys) - pad) / ocr_s)
                bx_max_l = min(float(box_crop_img.shape[1]), (max(xs) - pad) / ocr_s)
                by_max_l = min(float(box_crop_img.shape[0]), (max(ys) - pad) / ocr_s)

                # Map back to upscaled-image coords, then to original-image coords
                ux_min = (bx_min_l + b_xmin) / scale
                uy_min = (by_min_l + b_ymin) / scale
                ux_max = (bx_max_l + b_xmin) / scale
                uy_max = (by_max_l + b_ymin) / scale

                lx_min = max(0.0, min(float(pw), ux_min))
                ly_min = max(0.0, min(float(ph), uy_min))
                lx_max = max(0.0, min(float(pw), ux_max))
                ly_max = max(0.0, min(float(ph), uy_max))

                dgx_min = pgx_min + lx_min
                dgy_min = pgy_min + ly_min
                dgx_max = pgx_min + lx_max
                dgy_max = pgy_min + ly_max

                digit_det = Detection(
                    class_id=94,
                    class_name=f"digit_{clean}",
                    confidence=float(conf),
                    local_bbox=(float(lx_min - blx_min), float(ly_min - bly_min),
                                float(lx_max - blx_min), float(ly_max - bly_min)),
                    global_bbox=(float(dgx_min), float(dgy_min),
                                 float(dgx_max), float(dgy_max)),
                )
                if box_det.sub_detections is None:
                    box_det.sub_detections = []
                box_det.sub_detections.append(digit_det)

                # Also add flat reference to pole
                flat = Detection(
                    class_id=94,
                    class_name=f"digit_{clean}",
                    confidence=float(conf),
                    local_bbox=(float(lx_min), float(ly_min),
                                float(lx_max), float(ly_max)),
                    global_bbox=(float(dgx_min), float(dgy_min),
                                 float(dgx_max), float(dgy_max)),
                )
                if pole_det.sub_detections is None:
                    pole_det.sub_detections = []
                pole_det.sub_detections.append(flat)

        except Exception as e:
            logger.error(f"[OCR] Error on box_det {box_det.class_name}: {e}", exc_info=True)

    def execute_spatial_recursion(
        self,
        image: np.ndarray,
        root_bbox: List[float],
        flops_tracker: FlopsTracker
    ) -> List[Detection]:
        """
        Runs single-pass cascaded detection on the full image to detect traffic lights,
        lamp boxes, and LED countdown digits.
        """
        img_h, img_w = image.shape[:2]
        
        # Initialize custom detector if needed
        if self.custom_detector is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "yolov8_p2_custom_best.pt")
            self.custom_detector = SmallObjectDetector(model_path=model_path)

        logger.info("[Recursion Engine] Single-pass traffic light detection started.")
        
        # Detect traffic lights on the full image region using base model
        rx_min, ry_min, rx_max, ry_max = root_bbox
        rx_min, ry_min = max(0, int(round(rx_min))), max(0, int(round(ry_min)))
        rx_max, ry_max = min(img_w, int(round(rx_max))), min(img_h, int(round(ry_max)))
        
        root_crop = image[ry_min:ry_max, rx_min:rx_max]
        if root_crop.size == 0:
            return []
            
        p_boxes, p_confs, p_class_ids = yolo_service.run_inference(
            root_crop,
            conf_threshold=config.BASE_CONF_THRESHOLD,
            imgsz=config.BASE_IMGSZ,
        )
        flops_tracker.increment()
        
        pole_detections = []
        for p_box, p_conf, p_cls_id in zip(p_boxes, p_confs, p_class_ids):
            if p_cls_id == 9:  # Traffic light
                box_w = p_box[2] - p_box[0]
                box_h = p_box[3] - p_box[1]
                # Skip tiny noise boxes below the configured minimum size
                if box_w < config.MIN_BOX_WIDTH or box_h < config.MIN_BOX_HEIGHT:
                    continue
                gx_min = p_box[0] + rx_min
                gy_min = p_box[1] + ry_min
                gx_max = p_box[2] + rx_min
                gy_max = p_box[3] + ry_min
                
                pole_det = Detection(
                    class_id=9,
                    class_name="traffic light",  # Must be exactly "traffic light" for frontend titling
                    confidence=float(p_conf),
                    local_bbox=(float(p_box[0]), float(p_box[1]), float(p_box[2]), float(p_box[3])),
                    global_bbox=(float(gx_min), float(gy_min), float(gx_max), float(gy_max)),
                    sub_detections=[]
                )
                pole_detections.append(pole_det)
                
        if not pole_detections:
            logger.info("[Recursion Engine] No traffic lights found.")
            return []
            
        # Merge duplicate pole boxes using Weighted Box Fusion
        # Lower IoU threshold so overlapping boxes on the same light cluster merge correctly
        pole_detections = weighted_box_fusion(pole_detections, iou_threshold=config.WBF_IOU_POLE)
        logger.info(f"[Recursion Engine] Found {len(pole_detections)} unique traffic lights.")
        
        # Level 3: Crop and upscale each pole to find active lights and digits
        for p_idx, pole_det in enumerate(pole_detections):
            pgx_min, pgy_min, pgx_max, pgy_max = pole_det.global_bbox
            pw = pgx_max - pgx_min
            ph = pgy_max - pgy_min
            
            pole_crop = image[int(pgy_min):int(pgy_max), int(pgx_min):int(pgx_max)]
            if pole_crop.size == 0:
                continue
                
            # Adaptive upscale: tiny lamps get magnified more, large housings less.
            scale = config.compute_upscale_factor(pw, ph)
            upscaled = cv2.resize(
                pole_crop,
                (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LANCZOS4
            )

            # Optional low-light contrast enhancement (luminance-only, hue preserved).
            if config.CLAHE_ENABLED:
                upscaled = self._apply_clahe(upscaled)

            # Keep the HSV size filter physically consistent across upscale factors.
            hsv_min_area = config.HSV_MIN_CONTOUR_AREA * (scale / config.UPSCALE_FACTOR) ** 2
            
            box_detections = []
            
            # Run custom YOLO-P2
            if self.custom_detector is not None and self.custom_detector.model is not None:
                p2_dets = self.custom_detector.predict(upscaled, conf_threshold=config.CUSTOM_CONF_THRESHOLD)
                flops_tracker.increment()
                
                for p2_det in p2_dets:
                    c_name = p2_det["class_name"]
                    conf = p2_det["confidence"]
                    bbox = p2_det["bbox"]
                    
                    cls_id = 91 if c_name == 'red' else (92 if c_name == 'yellow' else 93)
                    color_name = "Red Light" if c_name == 'red' else ("Yellow Light" if c_name == 'yellow' else "Green Light")
                    
                    lx_min = bbox[0] / scale
                    ly_min = bbox[1] / scale
                    lx_max = bbox[2] / scale
                    ly_max = bbox[3] / scale
                    
                    gx_min = pgx_min + lx_min
                    gy_min = pgy_min + ly_min
                    gx_max = pgx_min + lx_max
                    gy_max = pgy_min + ly_max
                    
                    box_det = Detection(
                        class_id=cls_id,
                        class_name=color_name,
                        confidence=float(conf),
                        local_bbox=(float(lx_min), float(ly_min), float(lx_max), float(ly_max)),
                        global_bbox=(float(gx_min), float(gy_min), float(gx_max), float(gy_max)),
                        sub_detections=[]
                    )
                    box_detections.append(box_det)
                    
            # HSV fallback: green hue [35,110] covers cyan LEDs; sat>=35, val>=120 avoids vegetation noise
            hsv = cv2.cvtColor(upscaled, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array(config.HSV_RED1_LOWER)
            upper_red1 = np.array(config.HSV_RED1_UPPER)
            lower_red2 = np.array(config.HSV_RED2_LOWER)
            upper_red2 = np.array(config.HSV_RED2_UPPER)
            lower_yellow = np.array(config.HSV_YELLOW_LOWER)
            upper_yellow = np.array(config.HSV_YELLOW_UPPER)
            lower_green = np.array(config.HSV_GREEN_LOWER)
            upper_green = np.array(config.HSV_GREEN_UPPER)
            
            masks = {
                "Red Light": cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2)),
                "Yellow Light": cv2.inRange(hsv, lower_yellow, upper_yellow),
                "Green Light": cv2.inRange(hsv, lower_green, upper_green)
            }
            
            hsv_detections = []
            for color_name, mask in masks.items():
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > hsv_min_area:
                        x, y, cw, ch = cv2.boundingRect(cnt)
                        aspect_ratio = float(cw) / ch
                        if 0.4 <= aspect_ratio <= 2.5:
                            cls_id = 91 if color_name == "Red Light" else (92 if color_name == "Yellow Light" else 93)
                            
                            lx_min = x / scale
                            ly_min = y / scale
                            lx_max = (x + cw) / scale
                            ly_max = (y + ch) / scale
                            
                            gx_min = pgx_min + lx_min
                            gy_min = pgy_min + ly_min
                            gx_max = pgx_min + lx_max
                            sgy_max = pgy_min + ly_max
                            
                            hull = cv2.convexHull(cnt)
                            hull_area = cv2.contourArea(hull)
                            solidity = float(area) / hull_area if hull_area > 0 else 0.0
                            confidence = 0.5 + 0.5 * solidity
                            
                            hsv_det = Detection(
                                class_id=cls_id,
                                class_name=color_name,
                                confidence=float(confidence),
                                local_bbox=(float(lx_min), float(ly_min), float(lx_max), float(ly_max)),
                                global_bbox=(float(gx_min), float(gy_min), float(gx_max), float(sgy_max)),
                                sub_detections=[]
                            )
                            hsv_detections.append(hsv_det)
                            
            combined_boxes = box_detections + hsv_detections
            if combined_boxes:
                box_detections = weighted_box_fusion(combined_boxes, iou_threshold=config.WBF_IOU_LAMP)
                
            pole_det.sub_detections = box_detections
            
            # Level 4: OCR on active light box crops
            for box_det in box_detections:
                self._run_ocr_on_box(
                    box_det=box_det,
                    pole_det=pole_det,
                    upscaled=upscaled,
                    scale=scale,
                    pgx_min=pgx_min,
                    pgy_min=pgy_min,
                    pw=pw,
                    ph=ph,
                )
                        
        # ── Sign Filter ───────────────────────────────────────────────────────────
        filtered_pole_detections = []
        for pole_det in pole_detections:
            pgx_min, pgy_min, pgx_max, pgy_max = pole_det.global_bbox
            parent_w = pgx_max - pgx_min
            parent_h = pgy_max - pgy_min
            if parent_w <= 0 or parent_h <= 0:
                continue

            ar = parent_h / parent_w
            sub_dets = pole_det.sub_detections or []
            is_sign = False

            # Rule A — Tall-pole: very tall narrow box is a sign pole, not a light.
            if ar > config.SIGN_FILTER_TALL_AR and parent_w < config.SIGN_FILTER_TALL_MAX_W:
                logger.info(f"[Sign Filter] Rule A (tall-pole): {parent_w:.0f}×{parent_h:.0f} ar={ar:.2f}")
                is_sign = True

            # Rule B — Square-fill: square-ish box where colour sub covers most of it.
            if not is_sign:
                sq_range = config.SIGN_FILTER_SQ_AR_MIN <= ar <= config.SIGN_FILTER_SQ_AR_MAX
                in_width = config.SIGN_FILTER_SQ_MIN_W <= parent_w <= config.SIGN_FILTER_SQ_MAX_W
                if sq_range and in_width:
                    for sub in sub_dets:
                        if sub.class_id not in [91, 92, 93]:
                            continue
                        sgx_min, sgy_min, sgx_max, sgy_max = sub.global_bbox
                        sw, sh = sgx_max - sgx_min, sgy_max - sgy_min
                        w_r = sw / parent_w
                        h_r = sh / parent_h
                        a_r = (sw * sh) / (parent_w * parent_h)
                        if (w_r > config.SIGN_FILTER_SQ_FILL_RATIO and
                                h_r > config.SIGN_FILTER_SQ_FILL_RATIO and
                                a_r > config.SIGN_FILTER_SQ_AREA_RATIO):
                            logger.info(
                                f"[Sign Filter] Rule B (sq-fill): {parent_w:.0f}×{parent_h:.0f} "
                                f"ar={ar:.2f} sub={sub.class_name} w_r={w_r:.2f} h_r={h_r:.2f}")
                            is_sign = True
                            break

            # Rule C (new) — High-confidence sub with no spatial context:
            # If a detection has exactly 1 colour sub whose confidence-weighted
            # area is near the full parent and the parent itself has low confidence,
            # it's a mis-fired detection on a coloured background patch (e.g. a bus).
            if not is_sign and pole_det.confidence < config.GHOST_BOX_CONF_THRESHOLD:
                color_subs = [s for s in sub_dets if s.class_id in [91, 92, 93]]
                if len(color_subs) == 1:
                    sgx_min, sgy_min, sgx_max, sgy_max = color_subs[0].global_bbox
                    sw, sh = sgx_max - sgx_min, sgy_max - sgy_min
                    a_r = (sw * sh) / (parent_w * parent_h) if (parent_w * parent_h) > 0 else 0
                    if a_r > 0.50:
                        logger.info(
                            f"[Sign Filter] Rule C (low-conf bg patch): "
                            f"conf={pole_det.confidence:.2f} area_ratio={a_r:.2f}")
                        is_sign = True

            if not is_sign:
                filtered_pole_detections.append(pole_det)

        logger.info(
            f"[Sign Filter] {len(pole_detections) - len(filtered_pole_detections)} discarded, "
            f"{len(filtered_pole_detections)} kept.")

        # ── Post-process sub-detections ───────────────────────────────────────
        final_detections = []
        for pole_det in filtered_pole_detections:
            color_subs = [s for s in (pole_det.sub_detections or []) if s.class_id in [91, 92, 93]]
            other_subs = [s for s in (pole_det.sub_detections or []) if s.class_id not in [91, 92, 93]]

            # Ghost-box removal: no colour + low confidence
            if not color_subs and pole_det.confidence < config.GHOST_BOX_CONF_THRESHOLD:
                logger.info(
                    f"[Post-filter] Ghost box dropped (no colour, conf={pole_det.confidence:.2f})")
                continue

            # Deduplicate colour subs by configurable IoU
            deduped_color = []
            for cand in sorted(color_subs, key=lambda s: -s.confidence):
                is_dup = False
                for kept in deduped_color:
                    if kept.class_id != cand.class_id:
                        continue
                    b1, b2 = kept.global_bbox, cand.global_bbox
                    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
                    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                        iou = inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0
                        if iou > config.DEDUP_COLOR_IOU:
                            is_dup = True
                            break
                if not is_dup:
                    deduped_color.append(cand)

            pole_det.sub_detections = deduped_color + other_subs
            final_detections.append(pole_det)
        
        logger.info(f"[Post-filter] Final traffic light count: {len(final_detections)}")
        return final_detections

    def _single_pass_recursion(
        self,
        image: np.ndarray,
        global_bbox: List[float],
    ) -> List[Detection]:
        """
        Phase 5 single-pass pipeline using YOLO26-P2.
        Runs one full-frame inference, wraps results in Detection objects,
        applies Sign Filter and ghost-box removal.
        """
        img_h, img_w = image.shape[:2]
        rx_min, ry_min, rx_max, ry_max = (
            max(0, int(global_bbox[0])), max(0, int(global_bbox[1])),
            min(img_w, int(global_bbox[2])), min(img_h, int(global_bbox[3])),
        )
        roi = image[ry_min:ry_max, rx_min:rx_max]
        if roi.size == 0:
            return []

        raw = yolo26_engine.predict_full_frame(roi)
        if not raw:
            return []

        # Convert to Detection objects (no sub_detections at this stage — OCR optional)
        detections: List[Detection] = []
        for d in raw:
            x1, y1, x2, y2 = d["bbox"]
            detections.append(Detection(
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                local_bbox=(float(x1), float(y1), float(x2), float(y2)),
                global_bbox=(float(x1 + rx_min), float(y1 + ry_min),
                             float(x2 + rx_min), float(y2 + ry_min)),
                sub_detections=[],
            ))

        # Merge near-duplicate boxes (YOLO26 is NMS-free but P2+P3 can double-fire)
        detections = weighted_box_fusion(detections, iou_threshold=config.WBF_IOU_POLE)
        logger.info(f"[YOLO26 Single-Pass] {len(detections)} detections after WBF.")

        # Ghost-box removal (no colour sub → low conf)
        final = [d for d in detections
                 if d.confidence >= config.GHOST_BOX_CONF_THRESHOLD]
        logger.info(f"[YOLO26 Single-Pass] {len(final)} detections after ghost filter.")
        return final

    def process_context_aware_recursion(
        self,
        image: np.ndarray,
        global_bbox: List[float],
    ) -> List[Detection]:
        """
        Entry point for the detection pipeline.

        PIPELINE_MODE=recursive  (default) — crop+upscale+P2+HSV+OCR pipeline
        PIPELINE_MODE=single_pass          — YOLO26 full-frame single inference
                                             (requires yolo26p2_color_bstld.pt)
        """
        flops_tracker = FlopsTracker()
        try:
            if config.PIPELINE_MODE == "single_pass" and yolo26_engine.ready:
                logger.info("[Pipeline] Mode: single_pass (YOLO26-P2)")
                results = self._single_pass_recursion(image, global_bbox)
            else:
                if config.PIPELINE_MODE == "single_pass":
                    logger.warning("[Pipeline] single_pass requested but YOLO26 weights missing; "
                                   "falling back to recursive mode.")
                results = self.execute_spatial_recursion(image, global_bbox, flops_tracker)
        except Exception as e:
            logger.error(f"Error in pipeline: {e}", exc_info=True)
            results = []

        flops_tracker.log_metrics()
        return results

recursion_engine = RecursionEngine()
