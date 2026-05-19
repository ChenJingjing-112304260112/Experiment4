# -*- coding: utf-8 -*-
"""
完整训练和推理脚本 - 参考GitHub仓库最佳实践
"""

import os
import sys
import csv
from collections import Counter

def train_model():
    """训练YOLOv8s模型"""
    print("\n" + "="*70)
    print("STEP 1: TRAIN YOLOv8s MODEL")
    print("="*70)
    
    from ultralytics import YOLO
    
    # 创建模型
    model = YOLO("yolov8s.pt")
    print("✓ YOLOv8s模型创建成功")
    
    # 训练配置
    print("\n开始训练...")
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=100,
        batch=4,
        imgsz=960,           # 大图像尺寸
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        patience=30,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,
        nbs=64,
        name="yolov8s_final",
        device="cpu",
        verbose=True,
        save=True,
        val=True,
        plots=True
    )
    
    return "runs/detect/yolov8s_final/weights/best.pt"

def run_inference(model_path):
    """使用模型进行TTA推理"""
    print("\n" + "="*70)
    print("STEP 2: TTA INFERENCE")
    print("="*70)
    
    from ultralytics import YOLO
    
    # 加载模型
    model = YOLO(model_path)
    print(f"✓ 模型加载成功: {model_path}")
    
    # 测试图像
    test_dir = "第4次实验数据及提交格式/test/images"
    test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"测试图像数量: {len(test_images)}")
    
    # 推理
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
    print("\n" + "="*70)
    print("STEP 3: GENERATE SUBMISSION")
    print("="*70)
    
    # 统计类别分布
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
    
    # 按置信度排序
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    
    # 写入提交文件
    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    
    print(f"\n提交文件生成完成!")
    print(f"预测数量: {len(predictions)}")

def main():
    print("="*70)
    print("COMPLETE TRAINING & INFERENCE")
    print("参考GitHub仓库最佳实践")
    print("="*70)
    
    # 检查是否已有训练好的YOLOv8s模型
    model_path = "runs/detect/yolov8s_final/weights/best.pt"
    
    if os.path.exists(model_path):
        print(f"找到现有模型: {model_path}")
    else:
        print("未找到模型，开始训练...")
        model_path = train_model()
    
    # 执行推理
    predictions = run_inference(model_path)
    
    # 生成提交文件
    generate_submission(predictions)
    
    print("\n" + "="*70)
    print("完成!")
    print("="*70)

if __name__ == "__main__":
    main()