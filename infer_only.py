import csv
from pathlib import Path
from ultralytics import YOLO

print("Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully")

test_dir = Path("第4次实验数据及提交格式/test/images")
image_paths = sorted([p for p in test_dir.iterdir() if p.is_file()])
print(f"Found {len(image_paths)} test images")

output_path = Path("submission.csv")
with output_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
    )
    writer.writeheader()
    
    count = 0
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
            count += 1
    
    print(f"Total detections: {count}")

print(f"Submission file saved to: {output_path}")