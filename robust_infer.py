from ultralytics import YOLO
import csv
from pathlib import Path
import os

def main():
    model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    print("="*60)
    print("Traffic Sign Detection - Inference Script")
    print("="*60)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    print(f"\n[1/4] Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"Model loaded successfully")
    
    # 检查测试目录
    print(f"\n[2/4] Checking test directory: {test_dir}")
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"ERROR: Test directory not found: {test_dir}")
        return
    
    # 获取所有图片文件
    image_paths = sorted([p for p in test_path.iterdir() if p.is_file()])
    print(f"Found {len(image_paths)} test images")
    
    if len(image_paths) == 0:
        print("ERROR: No images found!")
        return
    
    print(f"\n[3/4] Running inference...")
    print(f"Confidence threshold: 0.001")
    print(f"Image size: 640")
    
    # 进行推理
    predictions = []
    for idx, img_path in enumerate(image_paths):
        results = model.predict(source=str(img_path), conf=0.001, imgsz=640, save=False, verbose=False)
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                predictions.append({
                    "image_id": img_path.name,
                    "class_id": int(box.cls[0].item()),
                    "x_center": box.xywhn[0][0].item(),
                    "y_center": box.xywhn[0][1].item(),
                    "width": box.xywhn[0][2].item(),
                    "height": box.xywhn[0][3].item(),
                    "confidence": box.conf[0].item(),
                })
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)} images, {len(predictions)} predictions so far")
    
    print(f"\n[4/4] Writing predictions to CSV...")
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    
    print(f"\n{'='*60}")
    print(f"COMPLETED!")
    print(f"  Total images processed: {len(image_paths)}")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Output file: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()