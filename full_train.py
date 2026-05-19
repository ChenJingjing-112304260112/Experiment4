
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ultralytics import YOLO
import time

print("STARTING FULL TRAINING")
print("="*60)

# 加载更大的模型
model = YOLO("yolov8l.pt")
print("Loaded YOLOv8l model")

# 开始训练
start = time.time()

results = model.train(
    data="第4次实验数据及提交格式/data.yaml",
    epochs=100,
    batch=2,
    imgsz=640,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    patience=30,
    augment=True,
    mosaic=1.0,
    mixup=0.4,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=30.0,
    translate=0.15,
    scale=0.5,
    shear=5.0,
    fliplr=0.5,
    name="full_training_run",
    device="cpu",
    verbose=True,
    save_period=10,
    val=True
)

elapsed = time.time() - start
print(f"Training done in {elapsed/60:.1f} minutes")
print(f"mAP@0.5: {results.box.map50:.4f}")

# 保存结果
with open("training_log.txt", "w") as f:
    f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")
    f.write(f"Time: {elapsed/60:.1f} minutes\n")

print("Results saved to training_log.txt")
