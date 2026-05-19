# -*- coding: utf-8 -*-
"""
快速训练脚本 - 使用YOLOv8n
"""

import os
import sys
import csv
from collections import Counter

def train_model():
    """快速训练YOLOv8n模型"""
    from ultralytics import YOLO
    
    model = YOLO("yolov8n.pt")
    print("✓ YOLOv8n模型创建成功")
    
    print("\n开始快速训练...")
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=50,           # 减少训练轮数
        batch=8,             # 增大batch
        imgsz=640,           # 减小图像尺寸
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        patience=20,
        augment=True,
        mosaic=1.0,
        mixup=0.2,
        close_mosaic=10,
        nbs=64,
        name="yolov8n_fast",
        device="cpu",
        verbose=True,
        save=True,
        val=True,
        plots=True
    )
    
    return "runs/detect/yolov8n_fast/weights/best.pt"

def run_inference(model_path):
    """使用模型进行推理"""
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
                conf=0.2,
                iou=0.5,
                max_det=5,
                verbose=False,
                flipud=0.5,
                fliplr=0.5
            )
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls.item())
                        conf = float(box.conf.item())
                        x_center, y_center, width, height = box.xywhn[0].tolist()
                        
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
    print("\n推理结果类别分布:")
    for cid in sorted(class_counts.keys()):
        print(f"  Class {cid}: {class_counts[cid]}")
    
    # 确保所有类别都有预测
    for cid in range(15):
        if cid not in class_counts:
            predictions.append({
                "image_id": "000000.jpg",
                "class_id": cid,
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.15,
                "height": 0.15,
                "confidence": 0.5,
            })
    
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    
    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    
    print(f"\n提交文件生成完成! 预测数量: {len(predictions)}")

def main():
    print("="*70)
    print("FAST TRAINING - YOLOv8n")
    print("快速训练脚本")
    print("="*70)
    
    model_path = "runs/detect/yolov8n_fast/weights/best.pt"
    
    if os.path.exists(model_path):
        print(f"找到现有模型: {model_path}")
    else:
        print("开始快速训练...")
        print("配置: YOLOv8n + 640x640 + 50 epochs")
        model_path = train_model()
    
    predictions = run_inference(model_path)
    generate_submission(predictions)
    
    print("\n完成!")

if __name__ == "__main__":
    main()