# -*- coding: utf-8 -*-
"""
创建纯英文路径的数据目录
解决YOLO无法处理中文路径的问题
"""

import os
import shutil

print("="*70)
print("CREATING ENGLISH PATH DATASET")
print("解决中文路径编码问题")
print("="*70)

base_dir = "第4次实验数据及提交格式"
output_dir = "traffic_signs_data"  # 纯英文路径

# 创建输出目录结构
os.makedirs(os.path.join(output_dir, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val", "labels"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test", "images"), exist_ok=True)

# 复制训练集
print("复制训练集...")
for img in os.listdir(os.path.join(base_dir, "train", "images")):
    shutil.copy(os.path.join(base_dir, "train", "images", img), os.path.join(output_dir, "train", "images", img))
for label in os.listdir(os.path.join(base_dir, "train", "labels")):
    shutil.copy(os.path.join(base_dir, "train", "labels", label), os.path.join(output_dir, "train", "labels", label))

# 复制验证集
print("复制验证集...")
for img in os.listdir(os.path.join(base_dir, "val", "images")):
    shutil.copy(os.path.join(base_dir, "val", "images", img), os.path.join(output_dir, "val", "images", img))
for label in os.listdir(os.path.join(base_dir, "val", "labels")):
    shutil.copy(os.path.join(base_dir, "val", "labels", label), os.path.join(output_dir, "val", "labels", label))

# 复制测试集
print("复制测试集...")
for img in os.listdir(os.path.join(base_dir, "test", "images")):
    shutil.copy(os.path.join(base_dir, "test", "images", img), os.path.join(output_dir, "test", "images", img))

# 创建data.yaml（使用相对路径）
data_yaml = """path: ./traffic_signs_data
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
print("英文路径数据创建完成!")
print("输出目录:", output_dir)
print("="*70)