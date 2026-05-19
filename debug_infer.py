from ultralytics import YOLO
import csv
from pathlib import Path

def main():
    model_path = "C:/Users/51273/Desktop/机器学习4/runs/detect/traffic_signs_complete/weights/best.pt"
    test_dir = "C:/Users/51273/Desktop/机器学习4/第4次实验数据及提交格式/test/images"
    output_path = "C:/Users/51273/Desktop/机器学习4/submission.csv"
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print("Model loaded successfully")
    
    print(f"\nScanning test directory: {test_dir}")
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    print(f"Found {len(image_paths)} test images")
    
    if len(image_paths) == 0:
        print("ERROR: No images found in test directory!")
        return
    
    print(f"\nFirst 5 images:")
    for p in image_paths[:5]:
        print(f"  - {p.name}")
    
    print(f"\nStarting inference with conf=0.001...")
    
    total_predictions = 0
    images_with_predictions = 0
    
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        
        for idx, result in enumerate(model.predict(source=[str(p) for p in image_paths], conf=0.001, save=False, verbose=False)):
            image_id = Path(result.path).name
            
            if result.boxes is not None and len(result.boxes) > 0:
                images_with_predictions += 1
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
                    total_predictions += 1
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(image_paths)} images...")
    
    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"  Images processed: {len(image_paths)}")
    print(f"  Images with predictions: {images_with_predictions}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Output file: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()