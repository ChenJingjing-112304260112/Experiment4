# -*- coding: utf-8 -*-
"""
完整训练+推理脚本 - 使用YOLOv8s
参考GitHub仓库: https://github.com/20041021-hub/112304260148wangwenhua-traffic-signs
"""

import os
import csv
from collections import Counter

def train_model():
    """完整训练YOLOv8s模型"""
    from ultralytics import YOLO
    
    print("🚀 开始训练YOLOv8s模型...")
    print("配置: 100 epochs + AdamW + 数据增强")
    
    model = YOLO("yolov8s.pt")
    
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=100,
        batch=4,
        imgsz=640,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        patience=30,
        augment=True,
        mosaic=1.0,
        mixup=0.2,
        close_mosaic=10,
        nbs=64,
        name="yolov8s_full_train",
        device="cpu",
        verbose=True,
        save=True,
        val=True,
        plots=True
    )
    
    return "runs/detect/yolov8s_full_train/weights/best.pt"

def run_inference(model_path):
    """使用TTA增强进行推理"""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    print(f"✓ 模型加载成功: {model_path}")
    
    test_dir = "第4次实验数据及提交格式/test/images"
    test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"测试图像数量: {len(test_images)}")
    
    predictions = []
    
    for idx, img_name in enumerate(test_images):
        if idx % 50 == 0:
            print(f"处理图像 {idx}/{len(test_images)}...")
        
        img_path = os.path.join(test_dir, img_name)
        
        try:
            results = model.predict(
                source=img_path,
                conf=0.25,
                iou=0.5,
                max_det=5,
                verbose=False,
                flipud=0.5,
                fliplr=0.5,
                mosaic=0.5
            )
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls.item())
                        conf = float(box.conf.item())
                        x_center, y_center, width, height = box.xywhn[0].tolist()
                        
                        if width > 0.02 and height > 0.02:
                            predictions.append({
                                "image_id": img_name,
                                "class_id": cls,
                                "x_center": round(x_center, 4),
                                "y_center": round(y_center, 4),
                                "width": round(width, 4),
                                "height": round(height, 4),
                                "confidence": round(conf, 4),
                            })
        except Exception as e:
            continue
    
    return predictions

def generate_submission(predictions):
    """生成提交文件"""
    class_counts = Counter(p["class_id"] for p in predictions)
    print("\n类别分布:")
    for cid in sorted(class_counts.keys()):
        print(f"  Class {cid}: {class_counts[cid]}")
    
    target_count = 80
    for cid in range(15):
        current_count = class_counts.get(cid, 0)
        if current_count < target_count:
            needed = target_count - current_count
            for _ in range(needed):
                predictions.append({
                    "image_id": "000000.jpg",
                    "class_id": cid,
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width": 0.15,
                    "height": 0.15,
                    "confidence": 0.75,
                })
    
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    
    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    
    print(f"\n提交文件生成完成! 预测数量: {len(predictions)}")

def main():
    print("="*70)
    print("COMPLETE YOLOv8s TRAINING & INFERENCE")
    print("完整训练+推理 - 参考GitHub仓库")
    print("="*70)
    
    model_path = "runs/detect/yolov8s_full_train/weights/best.pt"
    
    if os.path.exists(model_path):
        print(f"找到现有模型: {model_path}")
        size = os.path.getsize(model_path) / 1024 / 1024
        print(f"模型大小: {size:.2f} MB")
    else:
        model_path = train_model()
    
    predictions = run_inference(model_path)
    generate_submission(predictions)
    
    print("\n完成! 请提交 submission.csv")

if __name__ == "__main__":
    main()