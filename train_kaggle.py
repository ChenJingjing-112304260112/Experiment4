# -*- coding: utf-8 -*-
"""
基于Kaggle参考代码的完整训练脚本
参考: https://www.kaggle.com/code/pkdarabi/traffic-signs-detection-using-yolov8/notebook
"""

import subprocess
import sys
import os

print("="*70)
print("YOLOv8 TRAFFIC SIGNS TRAINING")
print("参考: Kaggle交通标志检测笔记本")
print("="*70)

# 创建完整的训练命令（基于Kaggle参考）
train_cmd = [
    sys.executable, "-m", "ultralytics", "train",
    "data=balanced_data/data.yaml",      # 使用平衡后的数据
    "model=yolov8n.pt",                 # 使用nano模型（参考代码使用）
    "epochs=100",                       # 参考建议: 10, 50, 100
    "batch=-1",                         # 自动检测最佳batch大小
    "imgsz=640",                        # 图像尺寸
    "optimizer=auto",                   # 自动选择优化器
    "lr0=0.001",                        # 初始学习率（参考建议）
    "lrf=0.01",
    "dropout=0.2",                      # dropout防止过拟合
    "cos_lr=True",                      # 余弦学习率衰减
    "patience=30",                      # 早停耐心值
    "augment=True",                     # 启用数据增强
    "mosaic=1.0",
    "mixup=0.3",
    "hsv_h=0.015",
    "hsv_s=0.7",
    "hsv_v=0.4",
    "degrees=10.0",
    "translate=0.1",
    "scale=0.5",
    "shear=2.0",
    "fliplr=0.5",
    "close_mosaic=10",                  # 训练后期关闭mosaic
    "freeze=10",                        # 冻结前10层进行迁移学习
    "name=yolov8_kaggle_train",
    "device=0",
    "verbose=True",
    "save=True",
    "val=True",
    "plots=True"
]

print("训练配置（基于Kaggle参考）:")
print("  模型: YOLOv8n")
print("  训练轮数: 100 epochs")
print("  Batch: 自动 (-1)")
print("  优化器: auto")
print("  学习率: 0.001")
print("  Dropout: 0.2")
print("  数据增强: mosaic + mixup + HSV + rotation")

print("\n" + "="*50)
print("开始训练...")
print("="*50)

# 执行训练
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