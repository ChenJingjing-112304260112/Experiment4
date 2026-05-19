# -*- coding: utf-8 -*-
import sys
import os

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Working directory:", os.getcwd())
print("="*60)

try:
    print("Step 1: Importing ultralytics...")
    from ultralytics import YOLO
    print("Step 2: Ultralytics imported successfully")
    
    print("Step 3: Loading model...")
    model = YOLO('yolov8s.pt')
    print("Step 4: Model loaded successfully")
    
    print("Step 5: Starting training...")
    print("This may take a long time...")
    
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=50,
        batch=4,
        imgsz=640,
        optimizer="AdamW",
        lr0=0.001,
        name="traffic_signs_proper_train",
        device="cpu",
        verbose=True,
        save=True
    )
    
    print("Step 6: Training completed!")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)