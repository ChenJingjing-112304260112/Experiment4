"""
最小化训练脚本 - 测试训练是否能进行
"""

from ultralytics import YOLO

print("Starting minimal training test...")

try:
    model = YOLO("yolov8s.pt")
    print("Model loaded successfully")
    
    results = model.train(
        data="combined_data.yaml",
        epochs=1,
        batch=2,
        imgsz=320,
        name="test_train",
        device="cpu",
        verbose=True
    )
    
    print("Training completed!")
    print(f"mAP@0.5: {results.box.map50}")
    
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()