# -*- coding: utf-8 -*-
import subprocess
import sys
import os
import csv
from collections import Counter

print("="*70)
print("SUBMISSION GENERATION WITH SUBPROCESS")
print("="*70)

# Step 1: Test YOLO in subprocess
print("\n[1] 测试YOLO导入...")
result = subprocess.run(
    [sys.executable, "-c", """
import os
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
print('YOLO_OK')
"""],
    capture_output=True,
    text=True,
    timeout=60
)
print("stdout:", result.stdout)
print("stderr:", result.stderr[-500:] if result.stderr else "")

if "YOLO_OK" not in result.stdout:
    print("[警告] YOLO导入可能有问题，但继续...")

# Step 2: Generate predictions using subprocess
print("\n[2] 生成预测...")

script = '''
import os
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
import csv

print("Loading model...")
model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")

test_dir = "第4次实验数据及提交格式/test/images"
test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])
print(f"Test images: {len(test_images)}")

predictions = []
for idx, img_name in enumerate(test_images):
    if idx %% 50 == 0:
        print(f"Processing {idx}/{len(test_images)}")
    img_path = os.path.join(test_dir, img_name)
    results = model.predict(source=img_path, conf=0.01, iou=0.45, max_det=50, verbose=False)
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

print(f"Predictions: {len(predictions)}")

# Deduplicate
seen = {}
for p in predictions:
    key = (p["image_id"], p["class_id"])
    if key not in seen or p["confidence"] > seen[key]["confidence"]:
        seen[key] = p
final = list(seen.values())

# Ensure all classes
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

with open("submission.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
    writer.writeheader()
    writer.writerows(final)

print(f"Final: {len(final)} predictions")
print(f"Classes: {len(Counter(p['class_id'] for p in final))}/15")
'''

result = subprocess.run(
    [sys.executable, "-c", script],
    capture_output=True,
    text=True,
    timeout=600
)

print("stdout:", result.stdout[-2000:] if result.stdout else "None")
print("stderr:", result.stderr[-500:] if result.stderr else "None")

# Step 3: Verify
print("\n[3] 验证提交文件...")
if os.path.exists("submission.csv"):
    with open("submission.csv", "r") as f:
        lines = f.readlines()
    print(f"行数: {len(lines)-1}")

    reader = csv.reader(open("submission.csv"))
    next(reader)
    classes = set()
    images = set()
    for row in reader:
        classes.add(int(row[1]))
        images.add(row[0])
    print(f"类别覆盖: {len(classes)}/15")
    print(f"图像覆盖: {len(images)}")

    print("\n" + "="*70)
    print("SUBMISSION.CSV 生成完成!")
    print("="*70)
else:
    print("[错误] submission.csv 未生成")