"""
创建优化的提交文件
使用置信度阈值0.05
"""

from ultralytics import YOLO
import csv
from pathlib import Path

# 加载模型
model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")

# 设置参数
test_dir = "第4次实验数据及提交格式/test/images"
output_path = "submission.csv"
conf_threshold = 0.05

# 获取测试图像
image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
print(f"Processing {len(image_paths)} test images with conf={conf_threshold}")

# 生成预测
predictions = []
for img_path in image_paths:
    results = model.predict(
        source=str(img_path),
        conf=conf_threshold,
        iou=0.45,
        imgsz=640,
        verbose=False,
        device='cpu'
    )
    
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

# 写入提交文件
with Path(output_path).open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
    writer.writeheader()
    writer.writerows(predictions)

# 统计
from collections import Counter
class_counts = Counter(p['class_id'] for p in predictions)
avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)

print(f"\nResults:")
print(f"Total predictions: {len(predictions)}")
print(f"Average confidence: {avg_conf:.4f}")
print("\nClass distribution:")
for cid in range(15):
    print(f"  Class {cid}: {class_counts.get(cid, 0)}")

# 检查缺失类别
missing = [cid for cid in range(15) if cid not in class_counts]
if missing:
    print(f"\nMissing classes: {missing}")
else:
    print("\nAll 15 classes covered!")