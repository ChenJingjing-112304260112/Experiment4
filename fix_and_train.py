# -*- coding: utf-8 -*-
"""
修复数据配置文件编码问题并重新训练
"""

import os

# 创建新的数据配置文件
data_yaml = '''
path: C:/Users/51273/Desktop/机器学习4/第4次实验数据及提交格式
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
'''

# 写入文件（使用UTF-8编码）
with open("第4次实验数据及提交格式/data.yaml", "w", encoding="utf-8") as f:
    f.write(data_yaml)

print("Data config file fixed!")

# 现在运行训练
from ultralytics import YOLO
import time

print("\nStarting training...")
model = YOLO("yolov8n.pt")

start = time.time()
results = model.train(
    data="第4次实验数据及提交格式/data.yaml",
    epochs=50,
    batch=4,
    imgsz=416,
    name="fixed_train",
    device="cpu",
    verbose=True
)
elapsed = time.time() - start

print(f"\nTraining completed in {elapsed/60:.1f} minutes")
print(f"mAP@0.5: {results.box.map50:.4f}")

# 保存结果
with open("training_result.txt", "w", encoding="utf-8") as f:
    f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
    f.write(f"Time: {elapsed/60:.1f} minutes\n")

print("Results saved to training_result.txt")