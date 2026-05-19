# -*- coding: utf-8 -*-
import subprocess
import sys
import os

print("Starting training via PowerShell...")

result = subprocess.run(
    [
        "powershell", "-Command",
        "cd 'C:\\Users\\51273\\Desktop\\机器学习4'; python -m ultralytics train data='第4次实验数据及提交格式/data.yaml' model=yolov8s.pt epochs=50 batch=4 imgsz=640 device=cpu name=traffic_signs_new_train"
    ],
    capture_output=True,
    text=True,
    timeout=7200  # 2小时超时
)

print("\nSTDOUT:")
print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)

print("\nSTDERR:")
print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

print("\nReturn code:", result.returncode)