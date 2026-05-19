"""
使用数据增强的训练脚本
"""

from ultralytics import YOLO

def train_with_augmentation():
    print("="*60)
    print("YOLOv8 Training with Augmentation")
    print("="*60)
    
    # 使用YOLOv8s模型
    model = YOLO('yolov8s.pt')
    
    # 训练参数 - 使用增强
    results = model.train(
        data='第4次实验数据及提交格式/data.yaml',
        epochs=50,
        batch=4,
        imgsz=640,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        augment=True,              # 启用数据增强
        hsv_h=0.015,              # HSV色调变化
        hsv_s=0.7,                # HSV饱和度变化  
        hsv_v=0.4,                # HSV明度变化
        degrees=0.0,              # 旋转角度
        translate=0.1,            # 平移
        scale=0.5,                # 缩放
        shear=0.0,                # 剪切
        perspective=0.0,          # 透视变换
        flipud=0.0,               # 上下翻转
        fliplr=0.5,               # 左右翻转
        mosaic=1.0,               # mosaic增强
        mixup=0.0,                # mixup增强
        copy_paste=0.0,           # 复制粘贴增强
        name='traffic_signs_augmented',
        verbose=True,
        device='cpu'
    )
    
    print("\nTraining completed!")
    print(f"Best model saved to: runs/detect/traffic_signs_augmented/weights/best.pt")
    
    # 验证
    metrics = model.val()
    print(f"\nmAP50: {metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    
    return model

def generate_submission_with_new_model():
    """使用新模型生成提交文件"""
    model_path = "runs/detect/traffic_signs_augmented/weights/best.pt"
    
    try:
        model = YOLO(model_path)
    except:
        print(f"Model not found at {model_path}")
        return
    
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
    
    print(f"Generated {len(predictions)} predictions")

if __name__ == "__main__":
    model = train_with_augmentation()
    generate_submission_with_new_model()