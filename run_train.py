import subprocess
import os
import sys

# 创建训练命令
cmd = [
    sys.executable, "-u", "-c",
    '''
import sys
sys.stdout = open("train_output.log", "w")
sys.stderr = open("train_error.log", "w")

print("Starting training...")
from ultralytics import YOLO
print("Ultralytics imported")

model = YOLO("yolov8n.pt")
print("Model loaded")

results = model.train(
    data="第4次实验数据及提交格式/data.yaml",
    epochs=1,
    batch=2,
    imgsz=320,
    name="debug_train",
    device="cpu",
    verbose=True
)

print("Training completed")
print(f"mAP@0.5: {results.box.map50}")
'''
]

subprocess.run(cmd, cwd=os.getcwd())

# 读取输出文件
print("Reading output...")
if os.path.exists("train_output.log"):
    with open("train_output.log", "r") as f:
        print(f.read())

if os.path.exists("train_error.log"):
    with open("train_error.log", "r") as f:
        error_content = f.read()
        if error_content:
            print("\nErrors:")
            print(error_content)