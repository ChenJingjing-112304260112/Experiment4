"""
改进的训练脚本
针对验证集性能差的问题进行优化
"""

from ultralytics import YOLO

def main():
    print("="*60)
    print("Improved Training for Traffic Sign Detection")
    print("="*60)
    
    # 使用YOLOv8s模型
    model = YOLO('yolov8s.pt')
    
    # 训练参数优化
    results = model.train(
        data='第4次实验数据及提交格式/data.yaml',
        epochs=100,           # 增加训练轮数
        batch=4,
        imgsz=640,
        optimizer='AdamW',
        lr0=0.0005,           # 降低初始学习率
        lrf=0.0001,           # 降低最终学习率
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=10,     # 增加热身轮数
        box=7.5,
        cls=1.0,              # 增加分类损失权重
        dfl=1.5,
        cos_lr=True,
        close_mosaic=50,      # 晚些关闭mosaic
        patience=30,          # 增加早停耐心值
        name='traffic_signs_improved',
        verbose=True,
        device='cpu',
        seed=42,
        deterministic=True,
        augment=True,         # 启用增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,         # 增加旋转
        translate=0.1,
        scale=0.5,
        fliplr=0.5
    )
    
    print("\nTraining completed!")
    
    # 验证
    metrics = model.val()
    print(f"\nFinal Validation Results:")
    print(f"  mAP50: {metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  Precision: {metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"  Recall: {metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
    
    # 生成提交文件
    generate_submission(model)

def generate_submission(model):
    """生成提交文件"""
    import csv
    from pathlib import Path
    
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    predictions = []
    
    for img_path in image_paths:
        results = model.predict(source=str(img_path), conf=0.15, iou=0.45, verbose=False)
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
    
    print(f"\nGenerated {len(predictions)} predictions")
    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
    print(f"Average confidence: {avg_conf:.4f}")
    
    # 上传到GitHub
    import subprocess
    subprocess.run(["git", "add", "submission.csv", "improved_training.py"])
    subprocess.run(["git", "commit", "-m", f"Update submission with improved model ({len(predictions)} preds, conf={avg_conf:.4f})"])
    subprocess.run(["git", "push", "origin", "main"])
    print("Files uploaded to GitHub!")

if __name__ == "__main__":
    main()