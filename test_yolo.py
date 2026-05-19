from ultralytics import YOLO
print("Ultralytics imported")

model = YOLO("yolov8n.pt")
print("Model loaded")

# Test with sample image
import cv2
import numpy as np

# Create a dummy image
img = np.zeros((640, 640, 3), dtype=np.uint8)
results = model.predict(img, verbose=False)
print(f"Prediction completed: {len(results[0].boxes)} boxes")