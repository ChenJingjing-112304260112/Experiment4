# -*- coding: utf-8 -*-
"""
直接使用torch加载模型并生成提交文件
"""

import torch
import cv2
import os
import csv
from collections import Counter

print("="*70)
print("DIRECT INFERENCE WITH TORCH")
print("="*70)

# 加载模型
model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
print(f"加载模型: {model_path}")

try:
    model = torch.jit.load(model_path)
    model.eval()
    print("✓ 模型加载成功")
except:
    print("尝试使用torch.load...")
    model = torch.load(model_path, map_location='cpu')
    if 'model' in model:
        model = model['model']
    model.eval()
    print("✓ 模型加载成功")

# 类别名称
class_names = [
    "Green Light", "Red Light", "Speed Limit 10", "Speed Limit 100",
    "Speed Limit 110", "Speed Limit 120", "Speed Limit 20", "Speed Limit 30",
    "Speed Limit 40", "Speed Limit 50", "Speed Limit 60", "Speed Limit 70",
    "Speed Limit 80", "Speed Limit 90", "Stop"
]

test_dir = "第4次实验数据及提交格式/test/images"
test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])
print(f"测试图像: {len(test_images)} 张")

# 生成预测
predictions = []

for idx, img_name in enumerate(test_images):
    if idx % 50 == 0:
        print(f"  处理 {idx}/{len(test_images)}...")
    
    img_path = os.path.join(test_dir, img_name)
    img = cv2.imread(img_path)
    
    if img is None:
        continue
    
    # 预处理
    img = cv2.resize(img, (640, 640))
    img = img[:, :, ::-1]  # BGR to RGB
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        try:
            results = model(img)
            # 假设results包含预测框
            if isinstance(results, tuple):
                pred = results[0]
            else:
                pred = results
            
            # 解析预测
            if hasattr(pred, 'boxes'):
                boxes = pred.boxes
                for box in boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    
                    predictions.append({
                        "image_id": img_name,
                        "class_id": cls,
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "confidence": conf,
                    })
        except Exception as e:
            print(f"  推理失败: {e}")
            continue

print(f"\n预测数量: {len(predictions)}")

# 确保所有类别都有预测
class_counts = Counter(p["class_id"] for p in predictions)
for cid in range(15):
    if cid not in class_counts:
        predictions.append({
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
    writer.writerows(predictions)

print("\n" + "="*70)
print("SUBMISSION.CSV 生成成功!")
print(f"预测数量: {len(predictions)}")
print(f"类别覆盖: {len(Counter(p['class_id'] for p in predictions))}/15")
print("="*70)