# -*- coding: utf-8 -*-
"""
直接使用原始数据目录训练
避免中文路径问题
"""

import os
import sys

print("="*70)
print("TRAINING WITH ORIGINAL DATA DIRECTORY")
print("直接使用原始数据目录")
print("="*70)

# 直接使用原始数据目录
base_dir = "第4次实验数据及提交格式"

# 检查数据目录
if not os.path.exists(base_dir):
    print(f"✗ 数据目录不存在: {base_dir}")
    sys.exit(1)

print(f"✓ 数据目录存在: {base_dir}")

# 读取原始data.yaml并更新路径
data_yaml_path = os.path.join(base_dir, "data.yaml")
if os.path.exists(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        content = f.read()
    print("✓ 找到原始data.yaml")
else:
    # 创建新的data.yaml
    content = f"""path: {os.path.abspath(base_dir)}
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
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✓ 创建了新的data.yaml")

# 检查验证集目录
val_images_path = os.path.join(base_dir, "val", "images")
print(f"✓ 验证集目录存在: {val_images_path}")
print(f"  验证图像数量: {len(os.listdir(val_images_path))}")

# 开始训练
print("\n开始训练...")
try:
    from ultralytics import YOLO
    
    model = YOLO("yolov8n.pt")
    print("✓ YOLO模型创建成功")
    
    # 使用原始数据目录的data.yaml
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        batch=4,
        imgsz=416,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        patience=20,
        augment=True,
        name="yolov8_direct_train",
        device="cpu",
        verbose=True,
        save=True,
        val=True,
        plots=True
    )
    
    print("\n训练完成!")
    print("模型保存位置: runs/detect/yolov8_direct_train/weights/best.pt")
    
except ImportError as e:
    print(f"✗ YOLO导入失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ 训练失败: {e}")
    sys.exit(1)