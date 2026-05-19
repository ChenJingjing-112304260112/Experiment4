from ultralytics import YOLO
import os
import csv
from pathlib import Path

def train_model():
    """训练YOLOv8模型"""
    print("Loading model...")
    model = YOLO('yolov8n.pt')
    
    print("Starting training...")
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=30,
        batch=2,
        imgsz=640,
        optimizer='auto',
        name='traffic_signs_improved',
        verbose=True
    )
    
    print("Training completed!")
    return model

def generate_submission(model_path, test_dir, output_path):
    """生成提交文件"""
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    print(f"Processing test images in {test_dir}")
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    print(f"Found {len(image_paths)} test images")
    
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        
        for result in model.predict(source=[str(p) for p in image_paths], conf=0.001, save=False, verbose=False):
            image_id = Path(result.path).name
            if result.boxes is None:
                continue
            for box in result.boxes:
                x_center, y_center, width, height = box.xywhn[0].tolist()
                writer.writerow({
                    "image_id": image_id,
                    "class_id": int(box.cls[0].item()),
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "confidence": float(box.conf[0].item()),
                })
    
    print(f"Submission file saved to {output_path}")

if __name__ == "__main__":
    try:
        # 训练模型
        # model = train_model()
        
        # 检查是否已有训练好的模型
        model_path = "runs/detect/traffic_signs_improved/weights/best.pt"
        
        if os.path.exists(model_path):
            print(f"Found trained model at {model_path}")
        else:
            print(f"Model not found at {model_path}, starting training...")
            model = train_model()
        
        # 生成提交文件
        generate_submission(model_path, "第4次实验数据及提交格式/test/images", "submission.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()