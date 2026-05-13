"""
优化的提交脚本 - 使用0.05置信度阈值

根据数据分析：
- 98.45%的预测置信度 < 0.1
- 0.51%的预测置信度在0.1-0.2之间
- 使用0.05阈值可以保留一些可能正确的预测
"""

from ultralytics import YOLO
import csv
from pathlib import Path
import sys

def main():
    model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    # 使用优化的置信度阈值
    CONF_THRESHOLD = 0.05
    
    print(f"Using confidence threshold: {CONF_THRESHOLD}", file=sys.stderr)
    
    model = YOLO(model_path)
    test_path = Path(test_dir)
    image_paths = sorted([p for p in test_path.iterdir() if p.is_file()])
    
    print(f"Processing {len(image_paths)} images...", file=sys.stderr)
    
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
                pred = {
                    "image_id": img_path.name,
                    "class_id": int(box.cls[0].item()),
                    "x_center": float(box.xywhn[0][0].item()),
                    "y_center": float(box.xywhn[0][1].item()),
                    "width": float(box.xywhn[0][2].item()),
                    "height": float(box.xywhn[0][3].item()),
                    "confidence": float(box.conf[0].item()),
                }
                all_predictions.append(pred)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} | Predictions: {len(all_predictions)}", file=sys.stderr)
    
    print(f"Total predictions: {len(all_predictions)}", file=sys.stderr)
    
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(all_predictions)
    
    print(f"Submission saved to {output_path}", file=sys.stderr)

if __name__ == "__main__":
    main()