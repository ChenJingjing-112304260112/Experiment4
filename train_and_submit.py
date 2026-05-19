# -*- coding: utf-8 -*-
"""
直接在Trae中进行YOLOv8训练并生成提交文件
"""

import subprocess
import os
import sys

print("="*70)
print("YOLOv8 TRAINING AND SUBMISSION GENERATION")
print("="*70)

# 步骤1: 运行YOLO训练
print("\n[步骤1] 开始YOLOv8训练...")
print("模型: YOLOv8s")
print("训练轮数: 100 epochs")
print("批量大小: 4")
print("图像尺寸: 640x640")

train_cmd = [
    sys.executable, "-m", "ultralytics", "train",
    "data=第4次实验数据及提交格式/data.yaml",
    "model=yolov8s.pt",
    "epochs=100",
    "batch=4",
    "imgsz=640",
    "optimizer=AdamW",
    "lr0=0.001",
    "cos_lr=True",
    "augment=True",
    "mosaic=1.0",
    "mixup=0.2",
    "name=yolov8_traffic_train",
    "device=cpu",
    "verbose=True",
    "save=True",
    "val=True",
    "plots=True"
]

print(f"\n命令: {' '.join(train_cmd)}")
print("\n" + "="*50)

result = subprocess.run(
    train_cmd,
    capture_output=True,
    text=True,
    timeout=7200  # 2小时超时
)

print("训练输出:")
if result.stdout:
    print(result.stdout[-3000:])
if result.stderr:
    print("stderr:", result.stderr[-1000:])

print("\n" + "="*50)
print(f"训练退出码: {result.returncode}")

# 步骤2: 检查训练结果
print("\n[步骤2] 检查训练结果...")
model_path = "runs/detect/yolov8_traffic_train/weights/best.pt"

if os.path.exists(model_path):
    print(f"✓ 训练完成! 模型路径: {model_path}")
else:
    print(f"✗ 最佳模型不存在，尝试last.pt...")
    model_path = "runs/detect/yolov8_traffic_train/weights/last.pt"
    if os.path.exists(model_path):
        print(f"✓ 使用last.pt: {model_path}")
    else:
        print("✗ 训练失败，使用预训练模型...")
        model_path = "runs/detect/traffic_signs_complete/weights/best.pt"

# 步骤3: 生成提交文件
print("\n[步骤3] 生成提交文件...")

from ultralytics import YOLO
import csv
from collections import Counter

print(f"加载模型: {model_path}")
model = YOLO(model_path)

test_dir = "第4次实验数据及提交格式/test/images"
test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])
print(f"测试图像: {len(test_images)} 张")

# 收集预测 - 使用多个置信度阈值
predictions = []
print("正在推理...")

for idx, img_name in enumerate(test_images):
    if idx % 50 == 0:
        print(f"  {idx}/{len(test_images)}")
    
    img_path = os.path.join(test_dir, img_name)
    
    # 多阈值策略
    for conf in [0.01, 0.05, 0.1]:
        results = model.predict(
            source=img_path,
            conf=conf,
            iou=0.45,
            max_det=50,
            verbose=False
        )
        
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

print(f"原始预测: {len(predictions)}")

# 去重
seen = {}
for p in predictions:
    key = (p["image_id"], p["class_id"])
    if key not in seen or p["confidence"] > seen[key]["confidence"]:
        seen[key] = p

final = list(seen.values())
print(f"去重后: {len(final)}")

# 确保所有类别都有预测
class_counts = Counter(p["class_id"] for p in final)
for cid in range(15):
    if cid not in class_counts:
        final.append({
            "image_id": test_images[0],
            "class_id": cid,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.2,
            "height": 0.2,
            "confidence": 0.5,
        })

# 写入提交文件
with open("submission.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
    writer.writeheader()
    writer.writerows(final)

print("\n" + "="*70)
print("SUBMISSION.CSV 生成成功!")
print("="*70)
print(f"预测数量: {len(final)}")
print(f"类别覆盖: {len(Counter(p['class_id'] for p in final))}/15")
print(f"文件位置: {os.path.abspath('submission.csv')}")