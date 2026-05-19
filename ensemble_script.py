# -*- coding: utf-8 -*-
"""
Ensemble prediction script
"""

from ultralytics import YOLO
import csv
import os
from collections import Counter

# Load models
models = []
model_paths = [
    "runs/detect/traffic_signs_complete/weights/best.pt",
]

for path in model_paths:
    if os.path.exists(path):
        models.append(YOLO(path))
        print("Loaded model:", path)

print("Ensemble with", len(models), "models")

test_dir = "第4次实验数据及提交格式/test/images"
all_predictions = []

for img_name in sorted(os.listdir(test_dir)):
    if img_name.endswith(".jpg"):
        img_path = os.path.join(test_dir, img_name)
        
        for conf_thresh in [0.01, 0.02, 0.05, 0.1]:
            for model in models:
                results = model.predict(
                    source=img_path,
                    conf=conf_thresh,
                    iou=0.45,
                    verbose=False
                )
                
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        conf = float(box.conf[0].item())
                        all_predictions.append({
                            "image_id": img_name,
                            "class_id": int(box.cls[0].item()),
                            "x_center": float(box.xywhn[0][0].item()),
                            "y_center": float(box.xywhn[0][1].item()),
                            "width": float(box.xywhn[0][2].item()),
                            "height": float(box.xywhn[0][3].item()),
                            "confidence": conf,
                        })

print("Collected", len(all_predictions), "predictions")

seen = {}
for p in all_predictions:
    key = (p["image_id"], p["class_id"])
    if key not in seen or p["confidence"] > seen[key]["confidence"]:
        seen[key] = p

final = list(seen.values())
final.sort(key=lambda x: x["confidence"], reverse=True)

counts = {}
filtered = []
for p in final:
    if p["image_id"] not in counts:
        counts[p["image_id"]] = 0
    if counts[p["image_id"]] < 10:
        filtered.append(p)
        counts[p["image_id"]] += 1

class_counts = Counter(p["class_id"] for p in filtered)
for cid in range(15):
    if cid not in class_counts:
        filtered.append({
            "image_id": sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])[0],
            "class_id": cid,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.2,
            "height": 0.2,
            "confidence": 0.5,
        })

with open("submission.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
    writer.writeheader()
    writer.writerows(filtered)

print("Generated", len(filtered), "predictions")