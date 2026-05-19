from ultralytics import YOLO
import csv
from pathlib import Path

def generate_submission(model_path, test_dir, output_path):
    """使用训练好的模型生成提交文件"""
    print(f"Loading trained model from {model_path}")
    model = YOLO(model_path)
    
    print(f"Processing test images in {test_dir}")
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    print(f"Found {len(image_paths)} test images")
    
    results_count = 0
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
                results_count += 1
    
    print(f"Submission file saved to {output_path}")
    print(f"Total predictions: {results_count}")

if __name__ == "__main__":
    model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    generate_submission(model_path, test_dir, output_path)