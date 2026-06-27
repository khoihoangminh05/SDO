import requests
import json
import os

url = "http://127.0.0.1:8000/api/v1/detect"
uploads_dir = "d:/LapTrinh/System/SDO/apps/orchestrator/uploads"

# Find a real image
matched_file = None
if os.path.exists(uploads_dir):
    for f in os.listdir(uploads_dir):
        if f.endswith(('.webp', '.jpg', '.jpeg', '.png')):
            matched_file = os.path.join(uploads_dir, f)
            break

if not matched_file:
    print("No uploads found!")
    exit(1)

print(f"Testing real image: {matched_file}")

payload = {
    "image_id": os.path.basename(matched_file).split('.')[0],
    "image_path": matched_file,
    "zoom_level": 0,
    "global_bbox": [0.0, 0.0, 1.0, 1.0] # full image
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        data = response.json()
        print("Zoom Level:", data.get("zoom_level"))
        print("Is Leaf:", data.get("is_leaf"))
        print("Children Count:", len(data.get("children") or []))
        
        detections = data.get("detections", [])
        print(f"\nDetections Count: {len(detections)}")
        for i, det in enumerate(detections):
            sub_dets = det.get("sub_detections", []) or []
            sub_desc = []
            for sd in sub_dets:
                sub_desc.append(f"{sd.get('class_name')} ({sd.get('confidence'):.2f})")
            print(f"  [{i}] {det.get('class_name')} ({det.get('confidence'):.2f}), Box: {[int(c) for c in det.get('global_bbox')]} -> Subs: {', '.join(sub_desc)}")
    else:
        print("Response:", response.text)
except Exception as e:
    print("Error:", e)
