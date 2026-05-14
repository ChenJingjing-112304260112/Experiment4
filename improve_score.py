"""
改进分数的完整方案
使用更低的置信度阈值获取更多预测
"""

from ultralytics import YOLO
import csv
from pathlib import Path
import subprocess

def main():
    print("="*70)
    print("IMPROVE SCORE - COMPLETE SOLUTION")
    print("="*70)
    
    # 使用现有模型
    model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    print(f"Processing {len(image_paths)} test images")
    
    # 使用非常低的置信度阈值来获取更多预测
    # 竞赛评分使用mAP@0.5，需要尽可能多的正确预测
    predictions = []
    
    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=0.001,  # 非常低的阈值
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
    
    print(f"\nGenerated {len(predictions)} predictions")
    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
    print(f"Average confidence: {avg_conf:.4f}")
    
    # 写入提交文件
    with Path(output_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    
    # 统计类别分布
    from collections import Counter
    class_counts = Counter(p['class_id'] for p in predictions)
    print("\nClass distribution:")
    for class_id in range(15):
        count = class_counts.get(class_id, 0)
        percentage = (count / len(predictions)) * 100 if predictions else 0
        print(f"  Class {class_id}: {count} ({percentage:.2f}%)")
    
    # 上传到GitHub
    print("\nUploading to GitHub...")
    subprocess.run(["git", "add", "submission.csv", "improve_score.py"])
    subprocess.run(["git", "commit", "-m", f"Improve score: {len(predictions)} predictions"])
    subprocess.run(["git", "push", "origin", "main"])
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"Submission file: {output_path}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Average confidence: {avg_conf:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()