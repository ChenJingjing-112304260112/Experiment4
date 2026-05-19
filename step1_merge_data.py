"""
分步训练脚本 - Step 1: 合并数据集
"""

import os
import shutil
from collections import Counter

print("="*70)
print("STEP 1: MERGING DATASETS")
print("="*70)

# 合并训练集和验证集
combined_dir = "combined_dataset"
train_images = "第4次实验数据及提交格式/train/images"
train_labels = "第4次实验数据及提交格式/train/labels"
val_images = "第4次实验数据及提交格式/val/images"
val_labels = "第4次实验数据及提交格式/val/labels"

# 删除旧目录
if os.path.exists(combined_dir):
    shutil.rmtree(combined_dir)

# 创建合并目录
os.makedirs(combined_dir, exist_ok=True)
os.makedirs(os.path.join(combined_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(combined_dir, "labels"), exist_ok=True)

# 复制训练数据
idx = 0
for f in os.listdir(train_images):
    if f.endswith('.jpg'):
        shutil.copy(os.path.join(train_images, f), os.path.join(combined_dir, "images", f))
        label_name = f.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(train_labels, label_name)):
            shutil.copy(os.path.join(train_labels, label_name), os.path.join(combined_dir, "labels", label_name))
        idx += 1

print(f"Copied {idx} training images")

# 复制验证数据（重命名避免冲突）
val_idx = 0
for f in os.listdir(val_images):
    if f.endswith('.jpg'):
        new_name = f"val_{val_idx}_{f}"
        shutil.copy(os.path.join(val_images, f), os.path.join(combined_dir, "images", new_name))
        label_name = f.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(val_labels, label_name)):
            new_label = f"val_{val_idx}_{label_name}"
            shutil.copy(os.path.join(val_labels, label_name), os.path.join(combined_dir, "labels", new_label))
        val_idx += 1

print(f"Copied {val_idx} validation images")
print(f"Total: {idx + val_idx} images")

# 统计类别分布
class_counts = Counter()
label_files = [f for f in os.listdir(os.path.join(combined_dir, "labels")) if f.endswith('.txt')]
for label_file in label_files:
    with open(os.path.join(combined_dir, "labels", label_file), 'r') as f:
        for line in f:
            if line.strip():
                class_counts[int(line.split()[0])] += 1

print("\nCombined dataset class distribution:")
for cid in range(15):
    count = class_counts.get(cid, 0)
    print(f"  Class {cid}: {count} samples")

# 创建数据配置文件
data_yaml = f"""
train: {combined_dir}/images
val: {combined_dir}/images
nc: 15
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
"""

with open("combined_data.yaml", "w") as f:
    f.write(data_yaml)

print("\nCombined data.yaml created!")
print("\nStep 1 completed!")