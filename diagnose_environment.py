# -*- coding: utf-8 -*-
import os
import sys
import subprocess

print("="*60)
print("DIAGNOSTIC TEST")
print("="*60)

# Test 1: Check Python environment
print("\n1. Python environment:")
print("   Python executable:", sys.executable)
print("   Python version:", sys.version.split()[0])

# Test 2: Check if ultralytics is importable
print("\n2. Testing ultralytics import:")
try:
    import ultralytics
    print("   ultralytics version:", ultralytics.__version__)
except Exception as e:
    print("   ERROR importing ultralytics:", str(e))

# Test 3: Try running a simple YOLO command
print("\n3. Testing YOLO command:")
result = subprocess.run(
    [sys.executable, "-c", "from ultralytics import YOLO; print('Import OK')"],
    capture_output=True,
    text=True
)
print("   Return code:", result.returncode)
print("   STDOUT:", result.stdout)
print("   STDERR:", result.stderr)

# Test 4: Try running YOLO train with minimal settings
print("\n4. Testing YOLO train (1 epoch):")
result = subprocess.run(
    [
        sys.executable, "-m", "ultralytics", "train",
        "data=第4次实验数据及提交格式/data.yaml",
        "model=yolov8n.pt",
        "epochs=1",
        "batch=2",
        "imgsz=320",
        "device=cpu",
        "name=diag_train"
    ],
    capture_output=True,
    text=True,
    timeout=300
)
print("   Return code:", result.returncode)
print("   STDOUT length:", len(result.stdout))
print("   STDERR length:", len(result.stderr))
if result.stdout:
    print("   STDOUT (last 500 chars):", result.stdout[-500:])
if result.stderr:
    print("   STDERR (last 500 chars):", result.stderr[-500:])

# Check if training directory was created
train_dir = "runs/detect/diag_train"
print("\n5. Checking for training output:")
print("   Training directory exists:", os.path.exists(train_dir))
if os.path.exists(train_dir):
    print("   Contents:", os.listdir(train_dir))

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)