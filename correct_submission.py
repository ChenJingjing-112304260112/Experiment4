"""
根据Kaggle Notebook的正确格式生成提交文件
- 使用 conf=0.5 的高置信度阈值
- 确保格式完全匹配
"""

from ultralytics import YOLO
import csv
from pathlib import Path

def generate_correct_submission():
    model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    print("="*60)
    print("Generating Correct Submission File")
    print("="*60)
    
    # 使用notebook中的参数：conf=0.5
    CONF_THRESHOLD = 0.5
    
    model = YOLO(model_path)
    test_path = Path(test_dir)
    image_paths = sorted([p for p in test_path.iterdir() if p.is_file()])
    
    print(f"Found {len(image_paths)} test images")
    print(f"Using confidence threshold: {CONF_THRESHOLD}")
    
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
    
    # 写入CSV
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(all_predictions)
    
    print(f"Submission saved to {output_path}")
    
    # 验证文件格式
    print("\n" + "="*60)
    print("Validating submission format...")
    print("="*60)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"Total rows (excluding header): {len(rows)}")
        
        if len(rows) > 0:
            print(f"\nFirst row:")
            for key, value in rows[0].items():
                print(f"  {key}: {value}")
            
            # 检查坐标范围
            valid_coords = True
            for row in rows[:10]:
                x, y, w, h = float(row['x_center']), float(row['y_center']), float(row['width']), float(row['height'])
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    valid_coords = False
                    print(f"Invalid coordinates: {row}")
            
            if valid_coords:
                print("\n✓ Coordinates are in valid range [0, 1]")
            else:
                print("\n✗ Some coordinates are out of range!")

if __name__ == "__main__":
    generate_correct_submission()