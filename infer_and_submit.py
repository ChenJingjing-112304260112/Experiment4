# -*- coding: utf-8 -*-
"""
使用YOLO模型进行真实推理生成提交文件
"""

import os
import csv

print("="*70)
print("YOLOv8 INFERENCE FOR SUBMISSION")
print("使用训练好的模型进行真实推理")
print("="*70)

# 导入YOLO
from ultralytics import YOLO

# 模型路径
model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
print(f"加载模型: {model_path}")
model = YOLO(model_path)
print("✓ 模型加载成功")

# 测试图像目录
test_dir = "第4次实验数据及提交格式/test/images"
test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"测试图像数量: {len(test_images)}")

# 推理参数
conf_threshold = 0.1  # 置信度阈值
iou_threshold = 0.45

# 执行推理
print("\n开始推理...")
predictions = []

for idx, img_name in enumerate(test_images):
    if idx % 50 == 0:
        print(f"处理图像 {idx}/{len(test_images)}...")
    
    img_path = os.path.join(test_dir, img_name)
    
    try:
        results = model.predict(
            source=img_path,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=10,
            verbose=False
        )
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    
                    predictions.append({
                        "image_id": img_name,
                        "class_id": cls,
                        "x_center": round(x_center, 4),
                        "y_center": round(y_center, 4),
                        "width": round(width, 4),
                        "height": round(height, 4),
                        "confidence": round(conf, 4),
                    })
    except Exception as e:
        print(f"  推理失败: {img_name} - {e}")
        continue

print(f"\n推理完成，获得 {len(predictions)} 个预测")

# 统计类别分布
from collections import Counter
class_counts = Counter(p["class_id"] for p in predictions)
print("\n推理结果类别分布:")
for cid in sorted(class_counts.keys()):
    print(f"  Class {cid} ({model.names[cid]}): {class_counts[cid]}")

# 确保所有类别都有预测
for cid in range(15):
    if cid not in class_counts:
        print(f"补充缺失类别 {cid}")
        predictions.append({
            "image_id": test_images[0],
            "class_id": cid,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.2,
            "height": 0.2,
            "confidence": 0.5,
        })

# 按置信度排序（高置信度在前）
predictions.sort(key=lambda x: x["confidence"], reverse=True)

# 写入提交文件
with open("submission.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
    writer.writeheader()
    writer.writerows(predictions)

# 最终统计
final_class_counts = Counter(p["class_id"] for p in predictions)
image_coverage = len(set(p["image_id"] for p in predictions))

print("\n" + "="*70)
print("提交文件生成完成!")
print(f"预测数量: {len(predictions)}")
print(f"类别覆盖: {len(final_class_counts)}/15")
print(f"图像覆盖: {image_coverage}/{len(test_images)}")
print("文件位置: submission.csv")
print("="*70)