from ultralytics import YOLO
import csv
from pathlib import Path

def generate_submission(model_path, test_dir, output_path):
    """使用训练好的模型生成提交文件（带调试信息）"""
    print(f"Loading trained model from {model_path}")
    model = YOLO(model_path)
    print(f"Model loaded successfully")
    
    print(f"Processing test images in {test_dir}")
    test_path = Path(test_dir)
    print(f"Test directory exists: {test_path.exists()}")
    
    image_paths = sorted([p for p in test_path.iterdir() if p.is_file()])
    print(f"Found {len(image_paths)} test images")
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    # 打印前5个图像路径
    print("First 5 image paths:")
    for p in image_paths[:5]:
        print(f"  {p.name}")
    
    results_count = 0
    empty_count = 0
    
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        
        for idx, result in enumerate(model.predict(source=[str(p) for p in image_paths], conf=0.001, save=False, verbose=False)):
            image_id = Path(result.path).name
            if result.boxes is None:
                empty_count += 1
                continue
            if len(result.boxes) == 0:
                empty_count += 1
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
                results_count += 1
            
            # 每处理20张图片打印一次进度
            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1}/{len(image_paths)} images, {results_count} predictions so far")
    
    print(f"\nSubmission file saved to {output_path}")
    print(f"Total predictions: {results_count}")
    print(f"Images with no detections: {empty_count}")

if __name__ == "__main__":
    model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    print("=" * 50)
    print("Starting inference with trained model")
    print("=" * 50)
    
    generate_submission(model_path, test_dir, output_path)
    
    print("=" * 50)
    print("Inference completed!")
    print("=" * 50)