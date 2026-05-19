"""
诊断脚本 - 检查数据集和训练环境
"""

from ultralytics import YOLO
import os

print("="*70)
print("DATASET DIAGNOSTICS")
print("="*70)

# 检查数据集目录
data_dir = "第4次实验数据及提交格式"

# 统计训练集
train_images = os.path.join(data_dir, "train", "images")
train_labels = os.path.join(data_dir, "train", "labels")
val_images = os.path.join(data_dir, "val", "images")
val_labels = os.path.join(data_dir, "val", "labels")

print("Dataset statistics:")
print(f"Train images: {len(os.listdir(train_images))}")
print(f"Train labels: {len(os.listdir(train_labels))}")
print(f"Val images: {len(os.listdir(val_images))}")
print(f"Val labels: {len(os.listdir(val_labels))}")

# 检查类别分布
from collections import Counter
label_counts = Counter()

for label_file in os.listdir(train_labels):
    if label_file.endswith('.txt'):
        with open(os.path.join(train_labels, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    label_counts[int(parts[0])] += 1

print("\nClass distribution in training set:")
for cid in range(15):
    print(f"  Class {cid:2d} ({YOLO('yolov8n.pt').names[cid]}): {label_counts.get(cid, 0)} samples")

# 测试数据加载
print("\nTesting data loading...")
try:
    model = YOLO("yolov8n.pt")
    print("Model loaded successfully")
    
    # 尝试加载数据配置
    import yaml
    with open(os.path.join(data_dir, "data.yaml"), 'r') as f:
        data_config = yaml.safe_load(f)
    print("Data config loaded successfully")
    print(f"  nc: {data_config['nc']}")
    print(f"  names: {list(data_config['names'].values())[:5]}...")
    
    # 尝试第一次训练迭代
    print("\nStarting test training (1 epoch)...")
    results = model.train(
        data=os.path.join(data_dir, "data.yaml"),
        epochs=1,
        batch=2,
        imgsz=320,
        name="diagnostic_train",
        device="cpu",
        verbose=True
    )
    
    print(f"\nTraining completed!")
    print(f"mAP@0.5: {results.box.map50}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()