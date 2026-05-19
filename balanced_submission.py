"""
平衡的提交文件生成
确保覆盖所有类别同时保持较高置信度
"""

from ultralytics import YOLO
import csv
import os
from collections import Counter

# 删除旧文件
if os.path.exists("submission.csv"):
    os.remove("submission.csv")

# 加载模型
model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")

test_dir = "第4次实验数据及提交格式/test/images"
image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])

# 使用较低阈值获取所有预测，然后按置信度排序
all_predictions = []
for img_name in image_files:
    img_path = os.path.join(test_dir, img_name)
    results = model.predict(source=img_path, conf=0.01, verbose=False)
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            all_predictions.append({
                "image_id": img_name,
                "class_id": int(box.cls[0].item()),
                "x_center": float(box.xywhn[0][0].item()),
                "y_center": float(box.xywhn[0][1].item()),
                "width": float(box.xywhn[0][2].item()),
                "height": float(box.xywhn[0][3].item()),
                "confidence": float(box.conf[0].item()),
            })

# 按置信度排序
all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

# 统计每个类别的预测数量
class_counts = Counter(p['class_id'] for p in all_predictions)
print("All predictions class distribution:")
for cid in range(15):
    print(f"  Class {cid}: {class_counts.get(cid, 0)}")

# 选择最佳预测：高置信度 + 覆盖所有类别
final_predictions = []
selected_per_image = {}
min_conf_threshold = 0.1

for pred in all_predictions:
    img_id = pred['image_id']
    conf = pred['confidence']
    
    # 每个图像最多保留5个预测
    if img_id not in selected_per_image:
        selected_per_image[img_id] = []
    
    if len(selected_per_image[img_id]) >= 5:
        continue
    
    # 高置信度预测优先
    if conf >= min_conf_threshold:
        final_predictions.append(pred)
        selected_per_image[img_id].append(pred)

# 写入提交文件
with open("submission.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
    writer.writeheader()
    writer.writerows(final_predictions)

# 统计最终结果
final_counts = Counter(p['class_id'] for p in final_predictions)
avg_conf = sum(p['confidence'] for p in final_predictions) / len(final_predictions)

print(f"\nFinal submission:")
print(f"Total predictions: {len(final_predictions)}")
print(f"Average confidence: {avg_conf:.4f}")
print("\nClass distribution:")
for cid in range(15):
    count = final_counts.get(cid, 0)
    print(f"  Class {cid}: {count}")

missing = [cid for cid in range(15) if cid not in final_counts]
if missing:
    print(f"\n⚠️  Missing classes: {missing}")
else:
    print("\n✅  All 15 classes covered!")