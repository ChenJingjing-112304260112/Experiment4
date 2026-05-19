# -*- coding: utf-8 -*-
import subprocess
import sys
import os

os.chdir("C:\\Users\\51273\\Desktop\\机器学习4")

print("Starting training with shell=True...")
print("="*60)

cmd = 'python -m ultralytics train data="第4次实验数据及提交格式/data.yaml" model=yolov8s.pt epochs=50 batch=4 imgsz=640 device=cpu name=traffic_signs_shell_train'

result = subprocess.run(
    cmd,
    shell=True,
    capture_output=True,
    text=True,
    timeout=7200
)

print("Return code:", result.returncode)
print("\nSTDOUT:")
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
print("\nSTDERR:")
print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)