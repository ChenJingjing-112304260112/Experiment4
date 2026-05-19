"""
简单测试最新模型
"""

import os
from pathlib import Path

# 检查最新模型
model_dir = "runs/detect/traffic_signs_comprehensive/weights"
if Path(model_dir).exists():
    for f in os.listdir(model_dir):
        if f.endswith('.pt'):
            print(f"Found model: {f}")
else:
    print("Model directory does not exist!")

# 手动测试推理
from ultralytics import YOLO
import csv

model = YOLO("runs/detect/traffic_signs_comprehensive/weights/best.pt")
print("Model loaded successfully")

# 测试一张图片
test_img = "第4次实验数据及提交格式/test/images/00000.jpg"
results = model.predict(source=test_img, conf=0.2, verbose=False)
boxes = results[0].boxes
print(f"Test image predictions: {len(boxes) if boxes else 0}")

# 生成提交文件
test_dir = "第4次实验数据及提交格式/test/images"
output_path = "submission_new.csv"

image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
predictions = []

for img_path in image_paths[:10]:  # 只处理前10张测试
    results = model.predict(source=str(img_path), conf=0.2, verbose=False)
    if results[0].boxes is not None:
        for box in results[0].boxes:
            predictions.append({
                "image_id": img_path.name,
                "class_id": int(box.cls[0].item()),
                "x_center": float(box.xywhn[0][0].item()),
                "y_center": float(box.xywhn[0][1].item()),
                "width": float(box.xywhn[0][2].item()),
                "height": float(box.xywhn[0][3].item()),
                "confidence": float(box.conf[0].item()),
            })

print(f"Generated {len(predictions)} predictions for first 10 images")

# 写入新文件
with Path(output_path).open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
    writer.writeheader()
    writer.writerows(predictions)

print(f"Test file saved to {output_path}")