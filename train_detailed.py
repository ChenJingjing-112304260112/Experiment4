import sys
import os
from ultralytics import YOLO

# Redirect stdout and stderr to a log file
log_file = open('training_detailed.log', 'w')
sys.stdout = log_file
sys.stderr = log_file

print("Python version:", sys.version)
print("Working directory:", os.getcwd())
print("\nLoading YOLOv8n model...")

try:
    model = YOLO("yolov8n.pt")
    print("Model loaded successfully")
    
    print("\nStarting training...")
    print("="*50)
    
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=5,
        batch=2,
        imgsz=416,
        name="traffic_signs_detailed",
        verbose=True,
        device='cpu',
        workers=0
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print("Results object type:", type(results))
    
    # Check if weights directory exists
    weights_dir = "runs/detect/traffic_signs_detailed/weights"
    print(f"\nChecking weights directory: {weights_dir}")
    if os.path.exists(weights_dir):
        files = os.listdir(weights_dir)
        print(f"Files in weights directory: {files}")
    else:
        print("Weights directory does not exist!")
        
except Exception as e:
    print("\nError during training:", str(e))
    import traceback
    traceback.print_exc()
    
finally:
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("Training log saved to training_detailed.log")