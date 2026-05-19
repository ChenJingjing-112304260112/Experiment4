from ultralytics import YOLO
import os
import csv
from pathlib import Path

def train_and_generate_submission():
    """根据notebook逻辑训练模型并生成提交文件"""
    
    print("="*60)
    print("Traffic Signs Detection - Training and Submission Generation")
    print("="*60)
    
    # 1. 加载预训练模型（参考notebook: Final_model = YOLO('yolov8n.pt')）
    print("\n[1/4] Loading YOLOv8n pretrained model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully")
    
    # 2. 训练模型（参考notebook的训练参数）
    print("\n[2/4] Starting training...")
    print("Training parameters:")
    print("  - epochs: 30")
    print("  - batch: -1 (auto)")
    print("  - optimizer: auto")
    print("  - imgsz: 640")
    print("  - data: 第4次实验数据及提交格式/data.yaml")
    
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=30,
        batch=-1,
        imgsz=640,
        optimizer='auto',
        name='notebook_style_training',
        verbose=True
    )
    
    # 3. 加载训练好的最佳模型（参考notebook: Valid_model = YOLO('.../best.pt')）
    print("\n[3/4] Loading best model from training...")
    best_model_path = "runs/detect/notebook_style_training/weights/best.pt"
    
    if not os.path.exists(best_model_path):
        print(f"Warning: {best_model_path} not found, using last.pt")
        best_model_path = "runs/detect/notebook_style_training/weights/last.pt"
    
    print(f"Loading model from: {best_model_path}")
    trained_model = YOLO(best_model_path)
    
    # 4. 生成提交文件（参考notebook的推理逻辑）
    print("\n[4/4] Generating submission.csv...")
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    print(f"Found {len(image_paths)} test images")
    
    # 参考notebook的推理参数
    CONF_THRESHOLD = 0.5  # notebook中使用conf=0.5
    
    prediction_count = 0
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        
        for idx, result in enumerate(trained_model.predict(
            source=[str(p) for p in image_paths],
            conf=CONF_THRESHOLD,
            save=False,
            verbose=False
        )):
            image_id = Path(result.path).name
            
            if result.boxes is None or len(result.boxes) == 0:
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
                prediction_count += 1
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(image_paths)} images, {prediction_count} predictions so far...")
    
    print(f"\n{'='*60}")
    print(f"Submission file saved to: {output_path}")
    print(f"Total predictions: {prediction_count}")
    print(f"{'='*60}")
    
    return output_path

if __name__ == "__main__":
    try:
        output_file = train_and_generate_submission()
        print(f"\nSuccess! Submission file generated: {output_file}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()