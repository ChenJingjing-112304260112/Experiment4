# -*- coding: utf-8 -*-
"""
创建平衡数据目录
"""

import os
import shutil
from collections import Counter

print("="*70)
print("CREATING BALANCED DATASET")
print("="*70)

base_dir = "第4次实验数据及提交格式"
output_dir = "balanced_data"

# 创建输出目录结构
os.makedirs(os.path.join(output_dir, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val", "labels"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test", "images"), exist_ok=True)

# 复制验证集
print("复制验证集...")
for img in os.listdir(os.path.join(base_dir, "val", "images")):
    shutil.copy(os.path.join(base_dir, "val", "images", img), os.path.join(output_dir, "val", "images", img))
    label = img.replace(".jpg", ".txt")
    if os.path.exists(os.path.join(base_dir, "val", "labels", label)):
        shutil.copy(os.path.join(base_dir, "val", "labels", label), os.path.join(output_dir, "val", "labels", label))

# 复制测试集
print("复制测试集...")
for img in os.listdir(os.path.join(base_dir, "test", "images")):
    shutil.copy(os.path.join(base_dir, "test", "images", img), os.path.join(output_dir, "test", "images", img))

# 处理训练集（平衡类别）
print("\n处理训练集...")
train_images = os.listdir(os.path.join(base_dir, "train", "images"))
train_labels = os.listdir(os.path.join(base_dir, "train", "labels"))

# 创建图像到标签的映射
img_to_label = {}
for label in train_labels:
    img_name = label.replace(".txt", ".jpg")
    img_to_label[img_name] = label

# 统计类别分布
label_dir = os.path.join(base_dir, "train", "labels")
class_counts = Counter()
img_classes = {}

for label_file in train_labels:
    img_name = label_file.replace(".txt", ".jpg")
    classes_in_img = []
    with open(os.path.join(label_dir, label_file), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cid = int(parts[0])
                class_counts[cid] += 1
                classes_in_img.append(cid)
    img_classes[img_name] = classes_in_img

print("原始类别分布:")
for cid in sorted(class_counts.keys()):
    print(f"  Class {cid}: {class_counts[cid]}")

# 目标采样数量
max_count = max(class_counts.values())
target_count = int(max_count * 0.6)
print(f"\n目标采样数量: {target_count}")

# 为每个类别收集图像
class_images = {}
for cid in range(15):
    class_images[cid] = []

for img_name, classes in img_classes.items():
    for cid in classes:
        if img_name not in class_images[cid]:
            class_images[cid].append(img_name)

# 过采样/欠采样
import random
selected_images = set()

for cid in range(15):
    images = class_images[cid]
    current_count = len(images)
    
    if current_count >= target_count:
        selected = random.sample(images, target_count)
    else:
        selected = []
        while len(selected) < target_count:
            selected.extend(random.sample(images, min(len(images), target_count - len(selected))))
    
    selected_images.update(selected)

print(f"\n选中图像数量: {len(selected_images)}")

# 复制选中的训练图像和标签
print("复制训练数据...")
for img_name in selected_images:
    shutil.copy(os.path.join(base_dir, "train", "images", img_name), os.path.join(output_dir, "train", "images", img_name))
    label_name = img_to_label.get(img_name, img_name.replace(".jpg", ".txt"))
    shutil.copy(os.path.join(base_dir, "train", "labels", label_name), os.path.join(output_dir, "train", "labels", label_name))

# 创建data.yaml
data_yaml = f"""path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
test: test/images
nc: 15
names:
  0: Green Light
  1: Red Light
  2: Speed Limit 10
  3: Speed Limit 100
  4: Speed Limit 110
  5: Speed Limit 120
  6: Speed Limit 20
  7: Speed Limit 30
  8: Speed Limit 40
  9: Speed Limit 50
  10: Speed Limit 60
  11: Speed Limit 70
  12: Speed Limit 80
  13: Speed Limit 90
  14: Stop
"""

with open(os.path.join(output_dir, "data.yaml"), "w") as f:
    f.write(data_yaml)

print("\n" + "="*70)
print("平衡数据创建完成!")
print("输出目录:", output_dir)
print("="*70)