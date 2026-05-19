"""
完整的模型训练脚本
确保训练足够的epoch
"""

from ultralytics import YOLO
import subprocess
import os

def main():
    print("="*70)
    print("FULL MODEL TRAINING")
    print("="*70)
    
    # 训练参数
    model_name = "yolov8s.pt"
    epochs = 100
    batch_size = 8
    imgsz = 640
    
    print("Training parameters:")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {imgsz}")
    
    # 创建训练命令
    cmd = [
        "python", "-m", "ultralytics",
        "train",
        f"data=第4次实验数据及提交格式/data.yaml",
        f"model={model_name}",
        f"epochs={epochs}",
        f"batch={batch_size}",
        f"imgsz={imgsz}",
        "optimizer=AdamW",
        "lr0=0.001",
        "cos_lr=True",
        "augment=True",
        "name=traffic_signs_full_train",
        "device=cpu"
    ]
    
    print("\nStarting training...")
    print("Command:", " ".join(cmd))
    print("\n" + "="*70)
    
    # 执行训练
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    
    print("\nTraining completed!")
    print("\nSTDOUT:")
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    
    if result.returncode != 0:
        print("\nSTDERR:")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return
    
    # 加载训练好的模型
    model = YOLO("runs/detect/traffic_signs_full_train/weights/best.pt")
    
    # 在验证集上评估
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    val_results = model.val(data="第4次实验数据及提交格式/data.yaml", verbose=False)
    print(f"mAP@0.5: {val_results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {val_results.box.map:.4f}")
    
    # 生成提交文件
    print("\n" + "="*70)
    print("GENERATING SUBMISSION")
    print("="*70)
    
    import csv
    from pathlib import Path
    
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    predictions = []
    
    for img_path in image_paths:
        results = model.predict(source=str(img_path), conf=0.05, verbose=False)
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
    
    print(f"Generated {len(predictions)} predictions")
    
    # 上传到GitHub
    subprocess.run(["git", "add", "submission.csv", "full_training.py"])
    subprocess.run(["git", "commit", "-m", f"Full training: {epochs} epochs, {len(predictions)} predictions"])
    subprocess.run(["git", "push", "origin", "main"])
    
    print("\nDone!")

if __name__ == "__main__":
    main()