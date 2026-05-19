"""
最终解决方案 - 完整训练和推理流程
"""

import subprocess
import sys

def train_and_infer():
    print("="*60)
    print("Final Solution - Complete Training & Inference")
    print("="*60)
    
    # 步骤1: 使用YOLOv8s进行完整训练
    print("\nStep 1: Starting training...")
    
    from ultralytics import YOLO
    model = YOLO('yolov8s.pt')
    
    # 训练参数
    results = model.train(
        data='第4次实验数据及提交格式/data.yaml',
        epochs=150,
        batch=2,
        imgsz=640,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        augment=True,
        name='final_traffic_signs',
        verbose=True,
        device='cpu',
        patience=30
    )
    
    # 步骤2: 验证模型
    print("\nStep 2: Validating model...")
    metrics = model.val()
    mAP50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
    print(f"Validation mAP50: {mAP50:.4f}")
    
    # 步骤3: 生成提交文件
    print("\nStep 3: Generating submission...")
    import csv
    from pathlib import Path
    
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    predictions = []
    
    for idx, img_path in enumerate(image_paths):
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            iou=0.45,
            imgsz=640,
            verbose=False,
            device='cpu'
        )
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0].item())
                if conf >= 0.25:
                    predictions.append({
                        "image_id": img_path.name,
                        "class_id": int(box.cls[0].item()),
                        "x_center": float(box.xywhn[0][0].item()),
                        "y_center": float(box.xywhn[0][1].item()),
                        "width": float(box.xywhn[0][2].item()),
                        "height": float(box.xywhn[0][3].item()),
                        "confidence": conf,
                    })
    
    with Path(output_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    
    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
    print(f"Generated {len(predictions)} predictions")
    print(f"Average confidence: {avg_conf:.4f}")
    
    # 步骤4: 上传到GitHub
    print("\nStep 4: Uploading to GitHub...")
    subprocess.run(["git", "add", "submission.csv", "final_solution.py"])
    subprocess.run(["git", "commit", "-m", f"Final solution: {len(predictions)} preds, mAP50={mAP50:.4f}"])
    subprocess.run(["git", "push", "origin", "main"])
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Submission file: submission.csv")
    print(f"Predictions: {len(predictions)}")
    print(f"Average confidence: {avg_conf:.4f}")
    print(f"Validation mAP50: {mAP50:.4f}")

if __name__ == "__main__":
    train_and_infer()