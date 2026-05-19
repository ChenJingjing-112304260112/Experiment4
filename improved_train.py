# -*- coding: utf-8 -*-
"""
参考GitHub代码的改进训练脚本
使用YOLOv8m + 大图像尺寸
"""

import os
import sys

print("="*70)
print("IMPROVED TRAINING - YOLOv8m")
print("参考GitHub仓库的改进训练脚本")
print("="*70)

try:
    from ultralytics import YOLO
    print("✓ 成功导入YOLO")
except ImportError:
    print("✗ 安装YOLO...")
    os.system("pip install ultralytics -q")
    from ultralytics import YOLO
    print("✓ YOLO安装成功")

# 创建模型 - 使用YOLOv8m
model = YOLO("yolov8m.pt")
print("✓ YOLOv8m模型创建成功")

# 训练配置
print("\n开始训练...")
print("配置:")
print("  模型: YOLOv8m (25.9M参数)")
print("  epochs: 100")
print("  batch: 4")
print("  imgsz: 960")
print("  optimizer: AdamW")

try:
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=100,
        batch=4,
        imgsz=960,           # 大图像尺寸
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        patience=30,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,
        nbs=64,
        name="yolov8m_traffic",
        device="cpu",
        verbose=True,
        save=True,
        val=True,
        plots=True
    )
    
    print("\n训练完成!")
    print("模型保存位置: runs/detect/yolov8m_traffic/weights/best.pt")
    
except Exception as e:
    print(f"\n训练失败: {e}")
    print("注意: YOLOv8m在CPU上训练会很慢，可以尝试使用GPU或切换回YOLOv8n")
    sys.exit(1)