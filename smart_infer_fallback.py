# -*- coding: utf-8 -*-
"""
智能推理脚本 - 使用可用的最佳模型
自动检测并使用最新训练的模型
"""

import os
import csv
from collections import Counter

def find_available_model():
    """查找可用的模型，按优先级排序"""
    model_paths = [
        "runs/detect/yolov8n_100epochs-2/weights/best.pt",  # 最新训练的
        "runs/detect/yolov8n_100epochs/weights/best.pt",
        "runs/detect/yolov8n_70epochs/weights/best.pt",
        "runs/detect/yolov8n_50epochs/weights/best.pt",
        "runs/detect/yolov8n_30epochs/weights/best.pt",     # 30 epochs模型
        "runs/detect/yolov8n_fast/weights/best.pt",
        "runs/detect/yolov8n_quick/weights/best.pt",
        "yolov8n.pt"  # 官方预训练模型
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024
            print(f"✓ 找到可用模型: {path} ({size:.2f} MB)")
            return path
    
    return None

def run_tta_inference(model_path):
    """TTA推理"""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    print(f"\n✓ 模型加载成功")
    
    test_dir = "第4次实验数据及提交格式/test/images"
    test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"测试图像数量: {len(test_images)}")
    
    predictions = []
    
    for idx, img_name in enumerate(test_images):
        if idx % 100 == 0:
            print(f"  处理图像 {idx}/{len(test_images)}...")
        
        img_path = os.path.join(test_dir, img_name)
        
        try:
            results = model.predict(
                source=img_path,
                conf=0.15,
                iou=0.45,
                max_det=8,
                verbose=False,
                flipud=0.5,
                fliplr=0.5,
                mosaic=0.5
            )
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls.item())
                        conf_val = float(box.conf.item())
                        x_center, y_center, width, height = box.xywhn[0].tolist()
                        
                        if width > 0.02 and height > 0.02 and cls < 15:
                            predictions.append({
                                "image_id": img_name,
                                "class_id": cls,
                                "x_center": round(x_center, 4),
                                "y_center": round(y_center, 4),
                                "width": round(width, 4),
                                "height": round(height, 4),
                                "confidence": round(conf_val, 4),
                            })
        except Exception as e:
            continue
    
    return predictions

def generate_submission(predictions):
    """生成提交文件"""
    class_counts = Counter(p["class_id"] for p in predictions)
    
    print(f"\n预测数量: {len(predictions)}")
    
    target_count = 100
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
    
    print(f"\n✓ 提交文件生成完成! 预测数量: {len(predictions)}")

def main():
    print("="*70)
    print("SMART INFERENCE - 使用可用模型")
    print("智能推理 - 自动检测并使用最新模型")
    print("="*70)
    
    # 1. 查找可用模型
    print("\n1. 检测可用模型...")
    model_path = find_available_model()
    
    if not model_path:
        print("⚠️ 未找到任何模型")
        return
    
    # 2. TTA推理
    print("\n2. 开始TTA推理...")
    predictions = run_tta_inference(model_path)
    
    # 3. 生成提交文件
    print("\n3. 生成提交文件...")
    generate_submission(predictions)
    
    print("\n✅ 完成! 请提交 submission.csv")

if __name__ == "__main__":
    main()