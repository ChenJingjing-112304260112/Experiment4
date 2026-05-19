import sys
print("Python version:", sys.version)
print("Step 1: import os")
import os
print("Step 2: import torch")
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("Step 3: from ultralytics import YOLO")
from ultralytics import YOLO
print("YOLO imported successfully")
print("Step 4: YOLO model creation")
model = YOLO("yolov8s.pt")
print("Model created")
print("Step 5: test prediction")
results = model.predict(source="第4次实验数据及提交格式/test/images", conf=0.1, verbose=True)
print("Prediction done, boxes:", len(results[0].boxes) if results[0].boxes is not None else 0)