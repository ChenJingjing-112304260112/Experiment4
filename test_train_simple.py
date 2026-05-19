# -*- coding: utf-8 -*-
import os
import sys

# 确保目录存在
os.makedirs("training_output", exist_ok=True)

# 直接运行YOLO训练
import subprocess
result = subprocess.run(
    [
        sys.executable, "-u", "-c",
        '''
print("Testing YOLO training...")
from ultralytics import YOLO
print("Import successful")

model = YOLO("yolov8n.pt")
print("Model loaded")

# 简单训练
results = model.train(
    data="第4次实验数据及提交格式/data.yaml",
    epochs=5,
    batch=2,
    imgsz=320,
    name="test_run_5",
    device="cpu"
)

print(f"Training complete! mAP@0.5: {results.box.map50}")
'''
    ],
    capture_output=True,
    text=True,
    cwd=os.getcwd()
)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)