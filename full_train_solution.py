# -*- coding: utf-8 -*-
"""
完整训练脚本 - 解决所有问题
1. 使用英文路径
2. 使用绝对路径
3. 使用CPU训练
4. 自动创建必要的目录和配置
"""

import os
import sys
import shutil
from collections import Counter

print("="*70)
print("COMPLETE YOLOv8 TRAINING SOLUTION")
print("一次性解决所有问题")
print("="*70)

# 1. 创建英文路径数据目录（如果不存在）
base_dir = "第4次实验数据及提交格式"
output_dir = "traffic_signs_data"

if not os.path.exists(output_dir):
    print("\n1. 创建英文路径数据目录...")
    os.makedirs(os.path.join(output_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "images"), exist_ok=True)
    
    # 复制训练集
    for img in os.listdir(os.path.join(base_dir, "train", "images")):
        shutil.copy(os.path.join(base_dir, "train", "images", img), os.path.join(output_dir, "train", "images", img))
    for label in os.listdir(os.path.join(base_dir, "train", "labels")):
        shutil.copy(os.path.join(base_dir, "train", "labels", label), os.path.join(output_dir, "train", "labels", label))
    
    # 复制验证集
    for img in os.listdir(os.path.join(base_dir, "val", "images")):
        shutil.copy(os.path.join(base_dir, "val", "images", img), os.path.join(output_dir, "val", "images", img))
    for label in os.listdir(os.path.join(base_dir, "val", "labels")):
        shutil.copy(os.path.join(base_dir, "val", "labels", label), os.path.join(output_dir, "val", "labels", label))
    
    # 复制测试集
    for img in os.listdir(os.path.join(base_dir, "test", "images")):
        shutil.copy(os.path.join(base_dir, "test", "images", img), os.path.join(output_dir, "test", "images", img))
    
    print("   ✓ 数据目录创建完成")
else:
    print("\n1. 数据目录已存在，跳过创建")

# 2. 更新data.yaml使用绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "traffic_signs_data")

data_yaml = f"""path: {data_dir}
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

with open(os.path.join(data_dir, "data.yaml"), "w") as f:
    f.write(data_yaml)
print("2. ✓ data.yaml已更新")

# 3. 导入YOLO并训练
print("\n3. 开始训练...")
try:
    from ultralytics import YOLO
    
    model = YOLO("yolov8n.pt")
    print("   ✓ YOLO模型创建成功")
    
    # 训练配置（解决所有问题）
    results = model.train(
        data=os.path.join(data_dir, "data.yaml"),
        epochs=50,           # 减少训练轮数加快速度
        batch=4,             # CPU使用小batch
        imgsz=416,           # 减小图像尺寸加快速度
        optimizer='AdamW',   # 使用AdamW优化器
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        patience=20,
        augment=True,
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        name="yolov8_traffic_final",
        device="cpu",        # 使用CPU
        verbose=True,
        save=True,
        val=True,
        plots=True
    )
    
    print("\n" + "="*70)
    print("训练完成!")
    print("模型保存位置: runs/detect/yolov8_traffic_final/weights/best.pt")
    print("="*70)
    
except ImportError as e:
    print(f"   ✗ YOLO导入失败: {e}")
    print("   请安装: pip install ultralytics")
    sys.exit(1)
except Exception as e:
    print(f"   ✗ 训练失败: {e}")
    sys.exit(1)