import sys
import os
import cv2

sys.path.append("d:/LapTrinh/System/SDO/apps/worker")

from app.services.yolo_service import yolo_service
from app.services.ocr_service import ocr_service
from app.services.recursion_engine import recursion_engine

yolo_service.initialize()
ocr_service.initialize()

uploads_dir = "d:/LapTrinh/System/SDO/apps/orchestrator/uploads"
files = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if f.endswith(('.webp', '.jpg', '.jpeg', '.png'))]

print(f"Found {len(files)} uploaded images.")

for f_path in files:
    print("=" * 60)
    print(f"Processing image: {os.path.basename(f_path)}")
    image = cv2.imread(f_path)
    if image is None:
        print("Failed to load image!")
        continue
    h, w = image.shape[:2]
    
    detections = recursion_engine.process_context_aware_recursion(image, global_bbox=[0.0, 0.0, float(w), float(h)])
    print(f"Detections found: {len(detections)}")
    for i, det in enumerate(detections):
        sub_dets = det.sub_detections or []
        sub_desc = []
        for sd in sub_dets:
            sub_desc.append(f"{sd.class_name} ({sd.confidence:.2f})")
        print(f"  [{i}] Conf: {det.confidence:.2f}, Box: {[int(c) for c in det.global_bbox]} -> Subs: {', '.join(sub_desc)}")
