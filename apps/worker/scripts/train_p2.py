import argparse
import sys
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train custom YOLO-P2 model for small object detection.")
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10, 
        help="Number of epochs to train (default 10)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640, 
        help="Input image size (default 640)"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=8, 
        help="Batch size (default 8)"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="coco8.yaml", 
        help="Dataset config YAML path (default coco8.yaml)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="Device to use for training (default cpu)"
    )
    args = parser.parse_args()

    # Get absolute path to the custom YOLO-P2 config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_config = os.path.join(current_dir, "..", "models", "yolov8_p2_custom.yaml")
    yaml_config = os.path.abspath(yaml_config)
    
    if not os.path.exists(yaml_config):
        print(f"Error: Custom model configuration not found at {yaml_config}")
        sys.exit(1)
        
    print(f"Loading custom model config from: {yaml_config}")
    
    # Load model from config (NOT pre-trained weights)
    model = YOLO(yaml_config)
    
    print("Successfully compiled custom YOLOv8-P2 model. Starting training...")
    
    try:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=0 if args.device == "cpu" else 4,
            verbose=True
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
