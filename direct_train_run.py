# -*- coding: utf-8 -*-
import subprocess
import sys

# 直接运行YOLO训练命令
cmd = [
    sys.executable, "-u", "-c",
    '''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ultralytics import YOLO
import time

print("STARTING TRAINING...")
model = YOLO("yolov8n.pt")

start = time.time()
results = model.train(
    data="第4次实验数据及提交格式/data.yaml",
    epochs=50,
    batch=4,
    imgsz=416,
    name="direct_train",
    device="cpu"
)
elapsed = time.time() - start

print(f"TRAINING DONE!")
print(f"Time: {elapsed/60:.1f} minutes")
print(f"mAP@0.5: {results.box.map50:.4f}")

# Save results
with open("train_result.txt", "w") as f:
    f.write(f"mAP@0.5: {results.box.map50:.4f}\\n")
'''
]

print("Running training...")
result = subprocess.run(cmd, capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print("\nReturn code:", result.returncode)