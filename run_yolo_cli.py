# -*- coding: utf-8 -*-
import subprocess
import sys

print("Starting YOLO training via command line...")
print("Command: python -m ultralytics train data=第4次实验数据及提交格式/data.yaml model=yolov8s.pt epochs=50 batch=4 imgsz=640 device=cpu")

result = subprocess.run(
    [
        sys.executable,
        "-m", "ultralytics",
        "train",
        "data=第4次实验数据及提交格式/data.yaml",
        "model=yolov8s.pt",
        "epochs=50",
        "batch=4",
        "imgsz=640",
        "device=cpu",
        "name=traffic_signs_cli_train"
    ],
    capture_output=True,
    text=True,
    cwd="C:\\Users\\51273\\Desktop\\机器学习4"
)

print("\nSTDOUT:")
print(result.stdout)

print("\nSTDERR:")
print(result.stderr)

print("\nReturn code:", result.returncode)