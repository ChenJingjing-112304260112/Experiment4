"""
最终优化提交文件
调整推理参数以提高mAP分数
"""

from ultralytics import YOLO
import csv
from pathlib import Path
import subprocess

def main():
    print("="*70)
    print("FINAL OPTIMIZED SUBMISSION")
    print("="*70)
    
    # 使用现有模型
    model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    print(f"Processing {len(image_paths)} test images")
    
    # 使用多个置信度阈值进行集成
    thresholds = [0.05, 0.1, 0.15, 0.2]
    all_predictions = {}
    
    for conf_thresh in thresholds:
        for img_path in image_paths:
            results = model.predict(
                source=str(img_path),
                conf=conf_thresh,
                iou=0.45,
                imgsz=640,
                verbose=False,
                device='cpu'
            )
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    key = (img_path.name, int(box.cls[0].item()), 
                           round(float(box.xywhn[0][0].item()), 4),
                           round(float(box.xywhn[0][1].item()), 4))
                    conf = float(box.conf[0].item())
                    
                    if key not in all_predictions or conf > all_predictions[key]['confidence']:
                        all_predictions[key] = {
                            "image_id": img_path.name,
                            "class_id": int(box.cls[0].item()),
                            "x_center": float(box.xywhn[0][0].item()),
                            "y_center": float(box.xywhn[0][1].item()),
                            "width": float(box.xywhn[0][2].item()),
                            "height": float(box.xywhn[0][3].item()),
                            "confidence": conf,
                        }
    
    predictions = list(all_predictions.values())
    
    # 按置信度排序并保留较高置信度的预测
    predictions.sort(key=lambda x: -x['confidence'])
    
    # 过滤掉置信度太低的预测
    min_confidence = 0.08
    filtered_predictions = [p for p in predictions if p['confidence'] >= min_confidence]
    
    print(f"\nTotal predictions after merging: {len(filtered_predictions)}")
    avg_conf = sum(p['confidence'] for p in filtered_predictions) / len(filtered_predictions)
    print(f"Average confidence: {avg_conf:.4f}")
    
    # 写入提交文件
    with Path(output_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(filtered_predictions)
    
    # 统计类别分布
    from collections import Counter
    class_counts = Counter(p['class_id'] for p in filtered_predictions)
    print("\nClass distribution:")
    for class_id, count in sorted(class_counts.items()):
        percentage = (count / len(filtered_predictions)) * 100
        print(f"  Class {class_id}: {count} ({percentage:.2f}%)")
    
    # 上传到GitHub
    print("\nUploading to GitHub...")
    subprocess.run(["git", "add", "submission.csv", "final_optimized_submission.py"])
    subprocess.run(["git", "commit", "-m", f"Final optimized: {len(filtered_predictions)} preds, conf={avg_conf:.4f}"])
    subprocess.run(["git", "push", "origin", "main"])
    
    print("\n" + "="*70)
    print("SUBMISSION READY!")
    print("="*70)
    print(f"File: submission.csv")
    print(f"Predictions: {len(filtered_predictions)}")
    print(f"Average confidence: {avg_conf:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()