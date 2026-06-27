import sys
import os
import cv2
import logging

sys.path.append("d:/LapTrinh/System/SDO/apps/worker")

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from app.services.yolo_service import yolo_service
from app.services.ocr_service import ocr_service
from app.services.recursion_engine import recursion_engine

yolo_service.initialize()
ocr_service.initialize()

image_path = "d:/LapTrinh/System/SDO/apps/orchestrator/uploads/2990a4a3-92b5-4419-b2da-d16006021dea.webp"
image = cv2.imread(image_path)
h, w = image.shape[:2]

print(f"\n--- Running recursion engine locally on {os.path.basename(image_path)} ---")
detections = recursion_engine.process_context_aware_recursion(image, global_bbox=[0.0, 0.0, float(w), float(h)])

print(f"\nRemaining Detections: {len(detections)}")
for i, det in enumerate(detections):
    print(f"[{i}] {det.class_name} ({det.confidence:.2f}), Box: {[int(c) for c in det.global_bbox]}")
    for sd in det.sub_detections or []:
        print(f"  - Sub: {sd.class_name} ({sd.confidence:.2f}), Box: {[int(c) for c in sd.global_bbox]}")
