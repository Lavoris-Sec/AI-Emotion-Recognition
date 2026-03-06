import ssl
import os
from ultralytics import YOLO

# Bypass SSL certificate error
ssl._create_default_https_context = ssl._create_unverified_context

# File paths
model_path = r'D:\emotions_project\models\yolo11s.pt'
data_config = r'D:\emotions_project\data.yaml'

def train():
    # Load model
    model = YOLO(model_path)

    # Start training
    model.train(
        data=data_config,
        epochs=100,      
        imgsz=224,       
        batch=-1,        # Auto-batch for RTX 3060 Ti 8GB VRAM
        device=0,        # Use GPU
        workers=4,       
        name='Emotions_v11_Final',
        exist_ok=True,
        mosaic=1.0       
    )

if __name__ == "__main__":
    train()