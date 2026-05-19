# -*- coding: utf-8 -*-
import os
import sys

print("Current directory:", os.getcwd())
print("Python:", sys.executable)
print("="*60)

# 切换到工作目录
os.chdir("C:\\Users\\51273\\Desktop\\机器学习4")

# 直接使用os.system运行命令
print("Running training command...")
exit_code = os.system('python -m ultralytics train data="第4次实验数据及提交格式/data.yaml" model=yolov8s.pt epochs=50 batch=4 imgsz=640 device=cpu name=traffic_signs_os_train')

print("Exit code:", exit_code)