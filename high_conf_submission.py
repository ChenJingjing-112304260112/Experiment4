"""
使用较高置信度阈值生成提交文件
过滤掉低质量预测，提高mAP分数
"""

from ultralytics import YOLO
import csv
from pathlib import Path
import subprocess

def main():
    print("="*70)
    print("HIGH CONFIDENCE SUBMISSION")
    print("="*70)
    
    model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    # 使用较高的置信度阈值
    conf_threshold = 0.2
    
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    predictions = []
    
    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=0.45,
            imgsz=640,
            verbose=False,
            device='cpu'
        )
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                predictions.append({
                    "image_id": img_path.name,
                    "class_id": int(box.cls[0].item()),
                    "x_center": float(box.xywhn[0][0].item()),
                    "y_center": float(box.xywhn[0][1].item()),
                    "width": float(box.xywhn[0][2].item()),
                    "height": float(box.xywhn[0][3].item()),
                    "confidence": float(box.conf[0].item()),
                })
    
    # 写入提交文件
    with Path(output_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    
    # 统计
    from collections import Counter
    class_counts = Counter(p['class_id'] for p in predictions)
    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions) if predictions else 0
    
    print(f"Results with conf_threshold={conf_threshold}:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Average confidence: {avg_conf:.4f}")
    print("\nClass distribution:")
    for cid in range(15):
        count = class_counts.get(cid, 0)
        print(f"  Class {cid}: {count}")
    
    # 检查缺失类别
    missing = [cid for cid in range(15) if cid not in class_counts]
    if missing:
        print(f"\n⚠️  Missing classes: {missing}")
    
    # 上传到GitHub
    subprocess.run(["git", "add", "submission.csv", "high_conf_submission.py"])
    subprocess.run(["git", "commit", "-m", f"High confidence submission: {len(predictions)} preds, conf={conf_threshold}"])
    subprocess.run(["git", "push", "origin", "main"])
    
    print("\nDone!")

if __name__ == "__main__":
    main()