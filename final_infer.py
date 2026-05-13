from ultralytics import YOLO
import csv
from pathlib import Path
import sys

def main():
    # 使用正确的路径
    model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    print("="*60)
    print("Traffic Sign Detection - Final Inference")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Test images: {test_dir}")
    print(f"Output: {output_path}")
    print()
    
    # 检查模型文件
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    print(f"Model file size: {model_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 检查测试目录
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"ERROR: Test directory not found: {test_dir}")
        sys.exit(1)
    
    # 加载模型
    print("\nLoading YOLOv8 model...")
    model = YOLO(model_path)
    print("Model loaded successfully")
    
    # 获取测试图片
    image_paths = sorted([p for p in test_path.iterdir() if p.is_file()])
    print(f"Found {len(image_paths)} test images")
    
    if len(image_paths) == 0:
        print("ERROR: No test images found!")
        sys.exit(1)
    
    # 推理
    print("\nRunning inference...")
    all_predictions = []
    
    for idx, img_path in enumerate(image_paths):
        results = model.predict(
            source=str(img_path),
            conf=0.001,
            imgsz=640,
            save=False,
            verbose=False,
            device='cpu'
        )
        
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                pred = {
                    "image_id": img_path.name,
                    "class_id": int(box.cls[0].item()),
                    "x_center": box.xywhn[0][0].item(),
                    "y_center": box.xywhn[0][1].item(),
                    "width": box.xywhn[0][2].item(),
                    "height": box.xywhn[0][3].item(),
                    "confidence": box.conf[0].item(),
                }
                all_predictions.append(pred)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)} images, {len(all_predictions)} predictions")
    
    # 写入CSV
    print(f"\nWriting {len(all_predictions)} predictions to {output_path}...")
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(all_predictions)
    
    # 验证输出文件
    output_file = Path(output_path)
    if output_file.exists():
        print(f"\nOutput file created successfully!")
        print(f"  File: {output_file}")
        print(f"  Size: {output_file.stat().st_size / 1024:.2f} KB")
        print(f"  Predictions: {len(all_predictions)}")
    else:
        print("\nERROR: Output file was not created!")
    
    print("\n" + "="*60)
    print("Inference completed!")
    print("="*60)

if __name__ == "__main__":
    main()