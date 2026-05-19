# -*- coding: utf-8 -*-
"""
YOLOv8训练脚本 - 目标: 显著提高mAP@0.5到0.9以上
"""

from ultralytics import YOLO
import time
import os

print("="*70)
print("YOLOv8 交通标志检测 - 训练开始")
print("="*70)

# 加载模型 - 使用YOLOv8s (比n大，比m小，适合CPU训练)
model = YOLO('yolov8s.pt')
print("模型加载成功: yolov8s.pt")

# 开始训练
print("\n开始训练...")
print("配置:")
print("  - 训练轮数: 50 epochs")
print("  - 批量大小: 4")
print("  - 图像尺寸: 640")
print("  - 优化器: AdamW")
print("  - 学习率: 0.001")
print("  - 数据增强: 启用")
print("="*70)

start_time = time.time()

# 训练模型
results = model.train(
    data="第4次实验数据及提交格式/data.yaml",
    epochs=50,
    batch=4,
    imgsz=640,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    cos_lr=True,
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
    name="traffic_signs_proper_train",
    device="cpu",
    verbose=True,
    save=True,
    save_period=10,
    val=True,
    plots=True
)

elapsed_time = time.time() - start_time

print("\n" + "="*70)
print("训练完成!")
print("="*70)
print(f"训练时间: {elapsed_time/60:.2f} 分钟")
print(f"mAP@0.5: {results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {results.box.map:.4f}")

# 保存训练结果
with open("proper_training_results.txt", "w", encoding="utf-8") as f:
    f.write(f"训练时间: {elapsed_time/60:.2f} 分钟\n")
    f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")
    f.write(f"Epochs: {results.epoch}\n")

print("\n训练结果已保存到 proper_training_results.txt")