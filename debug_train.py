import os
import sys

print("Python version:", sys.version)
print("Working directory:", os.getcwd())

from ultralytics import YOLO
print("Ultralytics version:", __import__('ultralytics').__version__)

# 设置绝对路径
data_yaml_path = os.path.abspath("第4次实验数据及提交格式/data.yaml")
print("Data YAML path:", data_yaml_path)
print("YAML exists:", os.path.exists(data_yaml_path))

# 加载模型
model = YOLO("yolov8n.pt")
print("Model loaded successfully")

# 尝试训练
print("\nStarting training...")
try:
    results = model.train(
        data=data_yaml_path,
        epochs=1,
        batch=2,
        imgsz=320,
        name="debug_train",
        device="cpu",
        verbose=True
    )
    print("\nTraining completed!")
    print("Results type:", type(results))
    print("mAP@0.5:", results.box.map50)
except Exception as e:
    print(f"\nTraining failed with error: {e}")
    import traceback
    traceback.print_exc()