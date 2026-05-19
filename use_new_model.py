"""
使用最新训练的模型生成提交文件
"""

from ultralytics import YOLO
import csv
from pathlib import Path

def main():
    # 尝试使用最新的综合训练模型
    model_paths = [
        "runs/detect/traffic_signs_comprehensive/weights/best.pt",
        "runs/detect/traffic_signs_final/weights/best.pt",
        "runs/detect/traffic_signs_complete/weights/best.pt"
    ]
    
    model = None
    for path in model_paths:
        if Path(path).exists():
            model = YOLO(path)
            print(f"Loaded model: {path}")
            break
    
    if model is None:
        print("No model found!")
        return
    
    # 生成提交文件
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    predictions = []
    
    for idx, img_path in enumerate(image_paths):
        results = model.predict(
            source=str(img_path),
            conf=0.2,
            iou=0.45,
            imgsz=640,
            verbose=False,
            device='cpu',
            augment=True
        )
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0].item())
                if conf >= 0.2:
                    predictions.append({
                        "image_id": img_path.name,
                        "class_id": int(box.cls[0].item()),
                        "x_center": float(box.xywhn[0][0].item()),
                        "y_center": float(box.xywhn[0][1].item()),
                        "width": float(box.xywhn[0][2].item()),
                        "height": float(box.xywhn[0][3].item()),
                        "confidence": conf,
                    })
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)}")
    
    with Path(output_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    
    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
    print(f"\nGenerated {len(predictions)} predictions")
    print(f"Average confidence: {avg_conf:.4f}")
    
    # 上传到GitHub
    import subprocess
    subprocess.run(["git", "add", "submission.csv"])
    subprocess.run(["git", "commit", "-m", f"Update with new model: {len(predictions)} preds, conf={avg_conf:.4f}"])
    subprocess.run(["git", "push", "origin", "main"])
    print("Uploaded to GitHub!")

if __name__ == "__main__":
    main()