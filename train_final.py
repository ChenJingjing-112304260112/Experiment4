# -*- coding: utf-8 -*-
"""
完整训练脚本 - 基于Kaggle参考代码
参考: https://www.kaggle.com/code/pkdarabi/traffic-signs-detection-using-yolov8/notebook
"""

import sys
import os

print("="*70)
print("YOLOv8 TRAFFIC SIGNS TRAINING")
print("参考: Kaggle交通标志检测笔记本")
print("="*70)

# 确保使用正确的工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 导入YOLO
try:
    from ultralytics import YOLO
    print("✓ 成功导入YOLO")
except ImportError as e:
    print(f"✗ 导入YOLO失败: {e}")
    print("请先安装ultralytics: pip install ultralytics")
    sys.exit(1)

# 获取当前脚本目录（使用英文路径）
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"脚本目录: {script_dir}")

# 创建模型（参考Kaggle代码使用yolov8n）
print("\n创建YOLO模型...")
model = YOLO("yolov8n.pt")
print("✓ 模型创建成功")

# 训练配置（基于Kaggle参考）
print("\n开始训练...")
print("配置:")
print("  数据: balanced_data/data.yaml")
print("  模型: YOLOv8n")
print("  训练轮数: 100 epochs")
print("  批量大小: 4")
print("  图像尺寸: 640")
print("  优化器: auto")
print("  学习率: 0.001")
print("  数据增强: 启用")

# 使用英文路径
data_path = os.path.join(script_dir, "traffic_signs_data", "data.yaml")
print(f"\n数据路径: {data_path}")
print(f"数据文件存在: {os.path.exists(data_path)}")

# 执行训练（使用Kaggle推荐的参数）
results = model.train(
    data=data_path,
    epochs=100,
    batch=4,  # CPU使用较小的batch大小
    imgsz=640,
    optimizer='auto',
    lr0=0.001,
    lrf=0.01,
    dropout=0.2,
    cos_lr=True,
    patience=30,
    augment=True,
    mosaic=1.0,
    mixup=0.3,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    fliplr=0.5,
    close_mosaic=10,
    freeze=10,
    name="yolov8_traffic_signs",
    device="cpu",  # 使用CPU训练
    verbose=True,
    save=True,
    val=True,
    plots=True
)

print("\n" + "="*70)
print("训练完成!")
print("模型保存位置: runs/detect/yolov8_traffic_signs/weights/best.pt")
print("="*70)