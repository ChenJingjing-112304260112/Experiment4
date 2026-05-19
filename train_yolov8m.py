"""
使用YOLOv8m模型进行训练
更大的模型可能获得更好的性能
"""

from ultralytics import YOLO

def main():
    print("="*60)
    print("Training YOLOv8m Model")
    print("="*60)
    
    # 使用更大的YOLOv8m模型
    model = YOLO('yolov8m.pt')
    print("Loaded YOLOv8m model")
    
    # 训练
    results = model.train(
        data='第4次实验数据及提交格式/data.yaml',
        epochs=30,
        batch=2,
        imgsz=640,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        augment=True,
        name='traffic_signs_yolov8m',
        verbose=True,
        device='cpu'
    )
    
    print("\nTraining completed!")
    
    # 验证
    metrics = model.val()
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
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
        results = model.predict(source=str(img_path), conf=0.2, verbose=False)
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

if __name__ == "__main__":
    main()