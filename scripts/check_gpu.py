import torch
from ultralytics import YOLO

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Running on: {torch.cuda.get_device_name(0)}")

# Check model loading
try:
    model = YOLO('yolo11n.pt')
    print("YOLO is ready to work!")
except Exception as e:
    print(f"Error loading model: {e}")

    