# -*- coding: utf-8 -*-
"""
完整训练脚本 - 使用平衡后的数据
"""

import os
import subprocess
import sys

print("="*70)
print("YOLOv8 TRAINING WITH BALANCED DATA")
print("="*70)

# 创建训练命令
train_cmd = [
    sys.executable, "-m", "ultralytics", "train",
    "data=balanced_data/data.yaml",
    "model=yolov8s.pt",
    "epochs=150",
    "batch=8",
    "imgsz=640",
    "optimizer=AdamW",
    "lr0=0.001",
    "lrf=0.01",
    "cos_lr=True",
    "patience=30",
    "augment=True",
    "mosaic=1.0",
    "mixup=0.3",
    "hsv_h=0.015",
    "hsv_s=0.7",
    "hsv_v=0.4",
    "degrees=15.0",
    "translate=0.15",
    "scale=0.5",
    "shear=2.0",
    "fliplr=0.5",
    "flipud=0.2",
    "name=yolov8_balanced_train",
    "device=cpu",
    "verbose=True",
    "save=True",
    "save_period=10",
    "val=True",
    "plots=True",
    "weights=[1.0, 1.0, 30.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0]"
]

print("训练配置:")
print("  数据: balanced_data/")
print("  模型: YOLOv8s")
print("  训练轮数: 150 epochs")
print("  批量大小: 8")
print("  图像尺寸: 640x640")
print("  类别权重: Class 2权重提升30倍")
print("  增强: mosaic + mixup + HSV + rotation + flip")

print("\n" + "="*50)
print("开始训练...")
print("="*50)

result = subprocess.run(
    train_cmd,
    capture_output=True,
    text=True,
    timeout=7200
)

print("\n训练输出:")
if result.stdout:
    print(result.stdout[-3000:])
if result.stderr:
    print("stderr:", result.stderr[-1000:])

print("\n" + "="*70)
print("训练完成!")
print("="*70)