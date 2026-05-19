"""
使用重新训练的模型生成提交文件（降低置信度阈值）
"""

from ultralytics import YOLO
import csv
from pathlib import Path

def main():
    model_path = "runs/detect/traffic_signs_retrained/weights/best.pt"
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    print("="*60)
    print("Using Retrained Model with Lower Confidence")
    print("="*60)
    print(f"Model: {model_path}")
    
    model = YOLO(model_path)
    print("Model loaded successfully")
    
    test_path = Path(test_dir)
    image_paths = sorted([p for p in test_path.iterdir() if p.is_file()])
    print(f"Found {len(image_paths)} test images")
    
    # 使用较低的置信度阈值
    CONF_THRESHOLD = 0.1
    
    all_predictions = []
    
    for idx, img_path in enumerate(image_paths):
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESHOLD,
            iou=0.45,
            imgsz=640,
            save=False,
            verbose=False,
            device='cpu'
        )
        
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                all_predictions.append({
                    "image_id": img_path.name,
                    "class_id": int(box.cls[0].item()),
                    "x_center": float(box.xywhn[0][0].item()),
                    "y_center": float(box.xywhn[0][1].item()),
                    "width": float(box.xywhn[0][2].item()),
                    "height": float(box.xywhn[0][3].item()),
                    "confidence": float(box.conf[0].item()),
                })
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images")
    
    print(f"\nTotal predictions: {len(all_predictions)}")
    
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(all_predictions)
    
    print(f"Submission saved to {output_path}")
    
    # 统计
    if all_predictions:
        avg_conf = sum(p['confidence'] for p in all_predictions) / len(all_predictions)
        print(f"Average confidence: {avg_conf:.4f}")
        
        from collections import Counter
        class_counts = Counter(p['class_id'] for p in all_predictions)
        print("\nClass distribution:")
        for class_id, count in sorted(class_counts.items()):
            percentage = (count / len(all_predictions)) * 100
            print(f"  Class {class_id}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    main()