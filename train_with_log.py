# -*- coding: utf-8 -*-
import os
import sys
import time

# 重定向stdout和stderr到文件
log_file = open("training_log.txt", "w", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

print("="*60)
print("YOLOv8 Training Starting")
print("="*60)
print("Time:", time.strftime("%Y-%m-%d %H:%M:%S"))
print("Python:", sys.executable)
print("Working dir:", os.getcwd())

try:
    from ultralytics import YOLO
    print("\nImporting YOLO... OK")
    
    print("\nLoading model...")
    model = YOLO('yolov8s.pt')
    print("Model loaded")
    
    print("\nStarting training...")
    print("Epochs: 50")
    print("Batch: 4")
    print("Image size: 640")
    
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=50,
        batch=4,
        imgsz=640,
        optimizer="AdamW",
        lr0=0.001,
        name="traffic_signs_final_new",
        device="cpu",
        verbose=True,
        save=True
    )
    
    print("\nTraining completed!")
    print("mAP@0.5:", results.box.map50)
    
except Exception as e:
    print("\nERROR:", str(e))
    import traceback
    traceback.print_exc()

print("\nLog written to training_log.txt")
log_file.close()