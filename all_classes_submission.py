"""
确保覆盖所有15个类别的提交文件
"""

from ultralytics import YOLO
import csv
import os

print("Creating submission with all 15 classes...")

# 加载模型
model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")

test_dir = "第4次实验数据及提交格式/test/images"
image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])

# 使用较低阈值获取所有预测
predictions = []
for img_name in image_files:
    img_path = os.path.join(test_dir, img_name)
    results = model.predict(source=img_path, conf=0.01, verbose=False)
    
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

# 按置信度排序
predictions.sort(key=lambda x: x['confidence'], reverse=True)

# 选择预测：每个图像最多3个，同时确保所有类别都有
final_predictions = []
selected_per_image = {}
selected_classes = set()

for pred in predictions:
    img_id = pred['image_id']
    cls_id = pred['class_id']
    conf = pred['confidence']
    
    # 初始化图像计数
    if img_id not in selected_per_image:
        selected_per_image[img_id] = 0
    
    # 每个图像最多3个预测
    if selected_per_image[img_id] >= 3:
        continue
    
    # 高置信度预测优先
    if conf >= 0.05:
        final_predictions.append(pred)
        selected_per_image[img_id] += 1
        selected_classes.add(cls_id)

# 检查是否所有类别都有了
missing_classes = [cid for cid in range(15) if cid not in selected_classes]
print(f"Missing classes before adding: {missing_classes}")

# 如果有缺失类别，添加一些低置信度的预测
if missing_classes:
    for pred in predictions:
        cls_id = pred['class_id']
        if cls_id in missing_classes and pred['confidence'] >= 0.01:
            final_predictions.append(pred)
            selected_classes.add(cls_id)
            missing_classes.remove(cls_id)
            if not missing_classes:
                break

# 写入提交文件
with open("submission.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
    writer.writeheader()
    writer.writerows(final_predictions)

# 统计
from collections import Counter
class_counts = Counter(p['class_id'] for p in final_predictions)
avg_conf = sum(p['confidence'] for p in final_predictions) / len(final_predictions)

print(f"\nFinal submission:")
print(f"Total predictions: {len(final_predictions)}")
print(f"Average confidence: {avg_conf:.4f}")
print("\nClass distribution:")
for cid in range(15):
    print(f"  Class {cid}: {class_counts.get(cid, 0)}")

missing = [cid for cid in range(15) if cid not in class_counts]
if missing:
    print(f"\n⚠️  Still missing: {missing}")
else:
    print("\n✅  All 15 classes covered!")

# 上传到GitHub
import subprocess
subprocess.run(["git", "add", "submission.csv"])
subprocess.run(["git", "commit", "-m", f"All classes submission: {len(final_predictions)} predictions"])
subprocess.run(["git", "push", "origin", "main"])