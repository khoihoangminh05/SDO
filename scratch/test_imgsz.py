import sys
import os
import cv2

sys.path.append("d:/LapTrinh/System/SDO/apps/worker")

from app.services.yolo_service import yolo_service

yolo_service.initialize()

image_path = "d:/LapTrinh/System/SDO/apps/orchestrator/uploads/2990a4a3-92b5-4419-b2da-d16006021dea.webp"
image = cv2.imread(image_path)
h, w = image.shape[:2]

print(f"Image resolution: {w}x{h}")

for imgsz in [640, 1280, 1920]:
    print("\n" + "=" * 50)
    print(f"Testing with imgsz={imgsz}, conf=0.05...")
    # Pad to square before prediction as in yolo_service
    padded_image, pad_x, pad_y = yolo_service.apply_letterbox(image)
    results = yolo_service.model.predict(source=padded_image, conf=0.05, imgsz=imgsz, verbose=False)
    
    count = 0
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id == 9: # Traffic Light
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                # Remap coordinates
                lx_min = max(0, xyxy[0] - pad_x)
                ly_min = max(0, xyxy[1] - pad_y)
                lx_max = min(w, xyxy[2] - pad_x)
                ly_max = min(h, xyxy[3] - pad_y)
                
                print(f"  [{count}] Conf: {conf:.3f}, BBox: {[int(lx_min), int(ly_min), int(lx_max), int(ly_max)]}")
                count += 1
    print(f"Total traffic lights detected: {count}")
