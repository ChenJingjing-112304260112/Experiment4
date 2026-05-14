"""
直接修复方案 - 解决低分数问题
"""

from ultralytics import YOLO
import csv
from pathlib import Path

def main():
    print("="*60)
    print("Direct Fix for Low Score")
    print("="*60)
    
    # 使用当前模型生成提交
    model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    print(f"Processing {len(image_paths)} test images...")
    
    # 使用较低的置信度阈值来获取更多预测
    predictions = []
    
    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=0.05,  # 更低的阈值获取更多预测
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
    
    # 上传到GitHub
    import subprocess
    subprocess.run(["git", "add", "submission.csv", "direct_fix.py"])
    subprocess.run(["git", "commit", "-m", f"Direct fix: {len(predictions)} predictions"])
    subprocess.run(["git", "push", "origin", "main"])
    
    print("\nDone! Uploaded to GitHub.")

if __name__ == "__main__":
    main()