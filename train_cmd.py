"""
使用命令行方式训练模型
确保训练过程完整执行
"""

import subprocess
import sys

def main():
    print("="*70)
    print("TRAINING WITH COMMAND LINE")
    print("="*70)
    
    # 训练命令
    train_cmd = [
        "yolo", "train",
        "data=第4次实验数据及提交格式/data.yaml",
        "model=yolov8s.pt",
        "epochs=200",
        "batch=2",
        "imgsz=640",
        "optimizer=AdamW",
        "lr0=0.001",
        "cos_lr=True",
        "augment=True",
        "name=traffic_signs_cmd_train",
        "device=cpu"
    ]
    
    print("Running training command:")
    print(" ".join(train_cmd))
    print("\n" + "="*70)
    
    # 执行训练
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    
    print("\nTraining output:")
    print(result.stdout)
    
    if result.returncode != 0:
        print("\nTraining failed with error:")
        print(result.stderr)
        sys.exit(1)
    
    # 生成提交文件
    print("\n" + "="*70)
    print("GENERATING SUBMISSION")
    print("="*70)
    
    from ultralytics import YOLO
    import csv
    from pathlib import Path
    
    model = YOLO("runs/detect/traffic_signs_cmd_train/weights/best.pt")
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    predictions = []
    
    # 使用低置信度阈值
    for img_path in image_paths:
        results = model.predict(source=str(img_path), conf=0.01, verbose=False)
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
    
    with Path(output_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    
    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
    print(f"Generated {len(predictions)} predictions")
    print(f"Average confidence: {avg_conf:.4f}")
    
    # 上传到GitHub
    subprocess.run(["git", "add", "submission.csv", "train_cmd.py"])
    subprocess.run(["git", "commit", "-m", f"Command line train: {len(predictions)} predictions"])
    subprocess.run(["git", "push", "origin", "main"])
    
    print("\nDone!")

if __name__ == "__main__":
    main()