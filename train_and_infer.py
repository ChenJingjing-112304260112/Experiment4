from ultralytics import YOLO
import csv
from pathlib import Path

def train_model():
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    print("Starting training...")
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=30,
        batch=4,
        imgsz=416,
        name="traffic_signs",
        augment=True,
        patience=10,
        verbose=True,
    )
    return results

def run_inference():
    print("Loading trained model...")
    model = YOLO("runs/detect/traffic_signs/weights/best.pt")
    
    test_dir = Path("第4次实验数据及提交格式/test/images")
    image_paths = sorted([p for p in test_dir.iterdir() if p.is_file()])
    
    output_path = Path("submission.csv")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()
        
        for result in model.predict(source=[str(p) for p in image_paths], conf=0.25, save=False, verbose=False):
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
    
    print(f"Submission file saved to: {output_path}")

if __name__ == "__main__":
    try:
        train_model()
        run_inference()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()