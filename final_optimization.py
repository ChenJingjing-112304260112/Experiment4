"""
最终优化方案 - 根据评分标准调整参数
mAP@0.5 需要预测框与真实框IoU >= 0.5
"""

from ultralytics import YOLO
import csv
from pathlib import Path
import subprocess

def main():
    print("="*70)
    print("FINAL OPTIMIZATION FOR mAP@0.5")
    print("="*70)
    
    model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    
    # 测试多个置信度阈值，找到最佳平衡点
    thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    
    print("Testing different confidence thresholds:")
    for conf_thresh in thresholds:
        predictions = []
        for img_path in image_paths:
            results = model.predict(source=str(img_path), conf=conf_thresh, iou=0.45, verbose=False)
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    predictions.append(float(box.conf[0].item()))
        
        if predictions:
            avg_conf = sum(predictions) / len(predictions)
            print(f"  conf={conf_thresh:.2f}: {len(predictions)} predictions, avg_conf={avg_conf:.4f}")
    
    # 选择最佳阈值 - 平衡预测数量和置信度
    best_thresh = 0.05
    print(f"\nSelected best threshold: {best_thresh}")
    
    # 生成最终提交
    predictions = []
    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=best_thresh,
            iou=0.45,
            imgsz=640,
            verbose=False,
            device='cpu'
        )
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0].item())
                predictions.append({
                    "image_id": img_path.name,
                    "class_id": int(box.cls[0].item()),
                    "x_center": float(box.xywhn[0][0].item()),
                    "y_center": float(box.xywhn[0][1].item()),
                    "width": float(box.xywhn[0][2].item()),
                    "height": float(box.xywhn[0][3].item()),
                    "confidence": conf,
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
    
    print(f"\nFinal Submission Statistics:")
    print(f"{'='*50}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Average confidence: {avg_conf:.4f}")
    print(f"Confidence threshold: {best_thresh}")
    print(f"{'='*50}")
    print("Class distribution:")
    for class_id in range(15):
        count = class_counts.get(class_id, 0)
        percentage = (count / len(predictions)) * 100 if predictions else 0
        print(f"  Class {class_id}: {count} ({percentage:.2f}%)")
    
    # 检查是否有缺失的类别
    missing_classes = [cid for cid in range(15) if cid not in class_counts]
    if missing_classes:
        print(f"\n⚠️  Missing classes: {missing_classes}")
    else:
        print("\n✅  All 15 classes covered!")
    
    # 上传到GitHub
    print("\nUploading to GitHub...")
    subprocess.run(["git", "add", "submission.csv", "final_optimization.py"])
    subprocess.run(["git", "commit", "-m", f"Final optimization: {len(predictions)} preds, conf={best_thresh}"])
    subprocess.run(["git", "push", "origin", "main"])
    
    print("\n" + "="*70)
    print("DONE! Submission ready for competition!")
    print("="*70)

if __name__ == "__main__":
    main()