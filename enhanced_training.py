# -*- coding: utf-8 -*-
"""
增强训练脚本 - 提高模型质量
"""

import os
import sys

print("="*70)
print("ENHANCED TRAINING SCRIPT")
print("增强训练脚本 - 提高模型质量")
print("="*70)

try:
    from ultralytics import YOLO
    print("✓ 成功导入YOLO")
except ImportError:
    print("✗ 安装YOLO...")
    os.system("pip install ultralytics -q")
    from ultralytics import YOLO
    print("✓ YOLO安装成功")

# 创建模型
model = YOLO("yolov8s.pt")  # 使用更大的模型
print("✓ YOLOv8s模型创建成功")

# 训练配置
print("\n开始增强训练...")
print("配置:")
print("  模型: YOLOv8s (更大的模型)")
print("  epochs: 200")
print("  batch: 4")
print("  imgsz: 640")
print("  optimizer: AdamW")
print("  数据增强: 增强模式")

try:
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=200,           # 增加训练轮数
        batch=4,
        imgsz=640,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        patience=50,
        augment=True,
        mosaic=1.0,
        mixup=0.5,
        cutmix=0.3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.15,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        flipud=0.5,
        erasing=0.4,
        name="yolov8s_enhanced",
        device="cpu",
        verbose=True,
        save=True,
        val=True,
        plots=True,
        dropout=0.2           # 添加dropout防止过拟合
    )
    
    print("\n训练完成!")
    print("模型保存位置: runs/detect/yolov8s_enhanced/weights/best.pt")
    
except Exception as e:
    print(f"\n训练失败: {e}")
    sys.exit(1)