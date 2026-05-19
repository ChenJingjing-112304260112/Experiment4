# -*- coding: utf-8 -*-
"""
数据预处理和类别平衡脚本
解决交通标志检测数据不平衡问题
"""

import os
import shutil
import random
from collections import Counter

print("="*70)
print("DATA PREPROCESSING FOR TRAFFIC SIGNS")
print("="*70)

# 配置
base_dir = "第4次实验数据及提交格式"
output_dir = "balanced_data"

# 创建输出目录
os.makedirs(os.path.join(output_dir, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val", "labels"), exist_ok=True)

# 读取训练数据
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
img_classes = {}  # 图像 -> 类别列表

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

# 目标采样数量（基于最大类别的70%）
max_count = max(class_counts.values())
target_count = int(max_count * 0.7)
print(f"\n目标采样数量: {target_count}")

# 为每个类别收集图像
class_images = {}
for cid in range(15):
    class_images[cid] = []

for img_name, classes in img_classes.items():
    for cid in classes:
        if img_name not in class_images[cid]:
            class_images[cid].append(img_name)

# 过采样策略
selected_images = set()

for cid in range(15):
    images = class_images[cid]
    current_count = class_counts[cid]
    
    if len(images) >= target_count:
        # 欠采样：随机选择目标数量
        selected = random.sample(images, target_count)
    else:
        # 过采样：重复采样直到达到目标数量
        selected = []
        while len(selected) < target_count:
            selected.extend(random.sample(images, min(len(images), target_count - len(selected))))
    
    selected_images.update(selected)
    print(f"Class {cid}: {len(images)} -> {len(set(selected))} (实际采样)")

print(f"\n总采样图像数: {len(selected_images)}")

# 复制图像和标签
print("\n复制文件...")
for img_name in selected_images:
    # 复制图像
    src_img = os.path.join(base_dir, "train", "images", img_name)
    dst_img = os.path.join(output_dir, "train", "images", img_name)
    shutil.copy(src_img, dst_img)
    
    # 复制标签
    label_name = img_to_label.get(img_name, img_name.replace(".jpg", ".txt"))
    src_label = os.path.join(base_dir, "train", "labels", label_name)
    dst_label = os.path.join(output_dir, "train", "labels", label_name)
    shutil.copy(src_label, dst_label)

# 复制验证集
print("\n复制验证集...")
for img_name in os.listdir(os.path.join(base_dir, "val", "images")):
    shutil.copy(
        os.path.join(base_dir, "val", "images", img_name),
        os.path.join(output_dir, "val", "images", img_name)
    )
    label_name = img_name.replace(".jpg", ".txt")
    shutil.copy(
        os.path.join(base_dir, "val", "labels", label_name),
        os.path.join(output_dir, "val", "labels", label_name)
    )

# 复制测试集
print("复制测试集...")
for img_name in os.listdir(os.path.join(base_dir, "test", "images")):
    shutil.copy(
        os.path.join(base_dir, "test", "images", img_name),
        os.path.join(output_dir, "test", "images", img_name)
    )

# 创建新的data.yaml
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
print("数据预处理完成!")
print("输出目录:", output_dir)
print("="*70)