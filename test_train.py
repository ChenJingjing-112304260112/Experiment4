import sys
from ultralytics import YOLO

print("Python version:", sys.version)
print("Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully")

print("\nStarting training with verbose output...")
try:
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=2,
        batch=2,
        imgsz=416,
        name="test_run",
        verbose=True,
        device='cpu'
    )
    print("\nTraining completed!")
    print("Results:", results)
except Exception as e:
    print("\nError during training:", str(e))
    import traceback
    traceback.print_exc()