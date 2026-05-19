"""
全面的训练脚本 - 解决mAP低的根本问题
"""

from ultralytics import YOLO
import os
import subprocess

def train_model():
    print("="*60)
    print("Comprehensive Training for Traffic Sign Detection")
    print("="*60)
    
    # 使用YOLOv8s预训练模型
    model = YOLO('yolov8s.pt')
    
    # 训练参数优化
    print("\nStarting training with optimized parameters...")
    results = model.train(
        data='第4次实验数据及提交格式/data.yaml',
        epochs=200,           # 大幅增加训练轮数
        batch=2,              # 减小batch size避免内存问题
        imgsz=640,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        warmup_epochs=5,
        box=5.0,
        cls=0.5,
        dfl=1.0,
        cos_lr=True,
        close_mosaic=10,
        patience=50,
        name='traffic_signs_comprehensive',
        verbose=True,
        device='cpu',
        seed=42,
        deterministic=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.05,
        scale=0.25,
        fliplr=0.5
    )
    
    return model

def validate_model(model):
    """验证模型性能"""
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60)
    
    metrics = model.val(data='第4次实验数据及提交格式/data.yaml')
    
    print(f"\nOverall Metrics:")
    print(f"  mAP50: {metrics.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"  Precision: {metrics.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"  Recall: {metrics.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
    
    return metrics

def generate_final_submission(model):
    """生成最终提交文件"""
    import csv
    from pathlib import Path
    
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    print("\n" + "="*60)
    print("Generating Final Submission")
    print("="*60)
    
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
            augment=True  # TTA
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
            print(f"  Processed {idx + 1}/{len(image_paths)}")
    
    with Path(output_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    
    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
    print(f"\nGenerated {len(predictions)} predictions")
    print(f"Average confidence: {avg_conf:.4f}")
    
    # 上传到GitHub
    subprocess.run(["git", "add", "submission.csv", "comprehensive_train.py"])
    subprocess.run(["git", "commit", "-m", f"Comprehensive training: {len(predictions)} preds, conf={avg_conf:.4f}"])
    subprocess.run(["git", "push", "origin", "main"])
    print("Uploaded to GitHub!")

if __name__ == "__main__":
    model = train_model()
    validate_model(model)
    generate_final_submission(model)