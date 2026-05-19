"""
最终修复脚本 - 创建有效的提交文件
"""

from ultralytics import YOLO
import csv
import os

print("Creating submission.csv...")

# 加载模型
model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")

test_dir = "第4次实验数据及提交格式/test/images"
image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])

# 使用较低阈值确保有预测
predictions = []
for img_name in image_files:
    img_path = os.path.join(test_dir, img_name)
    results = model.predict(source=img_path, conf=0.05, verbose=False)
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            predictions.append({
                "image_id": img_name,
                "class_id": int(box.cls[0].item()),
                "x_center": float(box.xywhn[0][0].item()),
                "y_center": float(box.xywhn[0][1].item()),
                "width": float(box.xywhn[0][2].item()),
                "height": float(box.xywhn[0][3].item()),
                "confidence": float(box.conf[0].item()),
            })

print(f"Found {len(predictions)} predictions")

# 写入提交文件
with open("submission.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
    writer.writeheader()
    writer.writerows(predictions)

print("submission.csv created successfully!")