"""
使用测试时增强(TTA)来改进预测质量
"""

from ultralytics import YOLO
import csv
from pathlib import Path

def main():
    model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    print("="*60)
    print("Inference with Test-Time Augmentation (TTA)")
    print("="*60)
    
    model = YOLO(model_path)
    
    test_path = Path(test_dir)
    image_paths = sorted([p for p in test_path.iterdir() if p.is_file()])
    print(f"Processing {len(image_paths)} test images...")
    
    all_predictions = []
    
    for idx, img_path in enumerate(image_paths):
        # 使用TTA进行预测
        results = model.predict(
            source=str(img_path),
            conf=0.1,
            iou=0.45,
            imgsz=640,
            save=False,
            verbose=False,
            device='cpu',
            augment=True,      # 启用TTA
            flipud=0.5,        # 上下翻转
            fliplr=0.5,        # 左右翻转
            mosaic=0.0,
            degrees=10.0,      # 旋转增强
            translate=0.1,     # 平移增强
            scale=0.9          # 缩放增强
        )
        
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0].item())
                # 只保留置信度较高的预测
                if conf >= 0.15:
                    all_predictions.append({
                        "image_id": img_path.name,
                        "class_id": int(box.cls[0].item()),
                        "x_center": float(box.xywhn[0][0].item()),
                        "y_center": float(box.xywhn[0][1].item()),
                        "width": float(box.xywhn[0][2].item()),
                        "height": float(box.xywhn[0][3].item()),
                        "confidence": conf,
                    })
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)} images")
    
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
    
    # 上传到GitHub
    import subprocess
    subprocess.run(["git", "add", "submission.csv", "infer_tta.py"])
    subprocess.run(["git", "commit", "-m", f"TTA inference: {len(all_predictions)} predictions, avg_conf={avg_conf:.4f}"])
    subprocess.run(["git", "push", "origin", "main"])
    print("\nFiles uploaded to GitHub!")

if __name__ == "__main__":
    main()