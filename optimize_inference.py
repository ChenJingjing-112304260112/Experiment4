"""
优化推理参数以提高分数
使用现有模型并调整参数
"""

from ultralytics import YOLO
import csv
from pathlib import Path
import subprocess

def main():
    print("="*70)
    print("OPTIMIZE INFERENCE FOR BETTER SCORE")
    print("="*70)
    
    # 使用现有模型
    model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    
    # 测试不同的置信度阈值
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    for conf_thresh in thresholds:
        predictions = []
        for img_path in image_paths:
            results = model.predict(source=str(img_path), conf=conf_thresh, verbose=False)
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
        
        avg_conf = sum(p['confidence'] for p in predictions) / len(predictions) if predictions else 0
        print(f"conf={conf_thresh}: {len(predictions)} predictions, avg_conf={avg_conf:.4f}")
    
    # 选择最佳阈值
    best_thresh = 0.01
    print(f"\nUsing best threshold: {best_thresh}")
    
    # 生成最终提交
    predictions = []
    for img_path in image_paths:
        results = model.predict(source=str(img_path), conf=best_thresh, verbose=False)
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
    
    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
    
    # 统计
    from collections import Counter
    class_counts = Counter(p['class_id'] for p in predictions)
    
    print(f"\nFinal submission stats:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Average confidence: {avg_conf:.4f}")
    print("\nClass distribution:")
    for class_id in range(15):
        count = class_counts.get(class_id, 0)
        print(f"  Class {class_id}: {count}")
    
    # 上传到GitHub
    subprocess.run(["git", "add", "submission.csv", "optimize_inference.py"])
    subprocess.run(["git", "commit", "-m", f"Optimize inference: {len(predictions)} predictions, conf={best_thresh}"])
    subprocess.run(["git", "push", "origin", "main"])
    
    print("\nDone!")

if __name__ == "__main__":
    main()