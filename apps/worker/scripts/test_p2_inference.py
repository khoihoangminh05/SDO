import os
import sys
import cv2

# Set path so Python can find app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.yolo_service import SmallObjectDetector

def main():
    # 1. Initialize detector with our custom weights
    weights_path = "yolov8_p2_custom_best.pt"
    print(f"Initializing SmallObjectDetector with: {weights_path}")
    detector = SmallObjectDetector(model_path=weights_path)
    
    # 2. Pick a test image
    test_img_path = "datasets/custom_dataset/images/train/10_png_jpg.rf.yznaeGnP4miPVnnC0ahw.jpg"
    print(f"Loading test image from: {test_img_path}")
    
    if not os.path.exists(test_img_path):
        # Fallback to any file in the directory if the specific one is missing
        img_dir = "datasets/custom_dataset/images/train"
        if os.path.exists(img_dir):
            files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
            if files:
                test_img_path = os.path.join(img_dir, files[0])
                print(f"Fallback to first available image: {test_img_path}")
            else:
                print("Error: No test images found in train folder!")
                return
        else:
            print("Error: Train dataset directory not found!")
            return

    image = cv2.imread(test_img_path)
    if image is None:
        print(f"Error: Could not read image at {test_img_path}")
        return
        
    print(f"Image loaded. Dimensions: {image.shape[1]}x{image.shape[0]}")
    
    # 3. Run predictions
    print("Running custom YOLO-P2 predictions...")
    detections = detector.predict(image, conf_threshold=0.25)
    
    print(f"\nDetections found: {len(detections)}")
    
    # 4. Draw bounding boxes
    for idx, det in enumerate(detections):
        cls_id = det["class_id"]
        cls_name = det["class_name"]
        conf = det["confidence"]
        bbox = det["bbox"] # [lx_min, ly_min, lx_max, ly_max]
        
        print(f"  [{idx}] Class: {cls_name} ({cls_id}), Conf: {conf:.4f}, BBox: {[int(c) for c in bbox]}")
        
        # Draw on image
        x1, y1, x2, y2 = [int(c) for c in bbox]
        # Colors: Red = (0, 0, 255), Yellow = (0, 255, 255), Green = (0, 255, 0) (BGR format)
        color = (0, 255, 0) # default green
        if "red" in cls_name.lower():
            color = (0, 0, 255)
        elif "yellow" in cls_name.lower():
            color = (0, 255, 255)
            
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image, 
            f"{cls_name} {conf:.2f}", 
            (x1, max(15, y1 - 5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color, 
            2
        )
        
    # Save output image
    out_dir = "runs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_p2_output.jpg")
    cv2.imwrite(out_path, image)
    print(f"\nSaved annotated test image output to: {out_path}")

if __name__ == "__main__":
    main()
