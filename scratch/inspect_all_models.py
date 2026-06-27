import os
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules

# Define a placeholder SPDConv so that torch.load can unpickle the model
class SPDConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1 * 4, c2, k, s, p if p is not None else k//2, groups=g, bias=False)
    def forward(self, x):
        return x

# Register it in tasks and modules
setattr(tasks, "SPDConv", SPDConv)
setattr(modules, "SPDConv", SPDConv)
setattr(modules.block, "SPDConv", SPDConv)

models_to_check = [
    "yolov8n.pt",
    "apps/worker/yolov8n.pt",
    "apps/worker/yolov8-visdrone.pt",
    "apps/worker/yolov8_p2_custom_best.pt",
    "apps/worker/runs/detect/train/weights/best.pt",
    "runs/detect/train/weights/best.pt"
]

for m_path in models_to_check:
    if os.path.exists(m_path):
        try:
            m = YOLO(m_path)
            print(f"PATH: {m_path}")
            print(f"CLASSES: {m.names}")
            print("-" * 50)
        except Exception as e:
            print(f"PATH: {m_path} - FAILED: {e}")
            print("-" * 50)
    else:
        print(f"PATH: {m_path} - NOT FOUND")
        print("-" * 50)
