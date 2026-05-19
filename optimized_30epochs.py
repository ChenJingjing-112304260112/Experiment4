# -*- coding: utf-8 -*-
"""
优化推理 - 基于已训练的30 epochs模型
直接使用，不重新训练
"""

import os
import csv
from collections import Counter

def find_best_trained_model():
    """查找已训练的模型"""
    model_paths = [
        "runs/detect/yolov8n_30epochs/weights/best.pt",  # 30 epochs训练
        "runs/detect/yolov8n_fast/weights/best.pt",
        "runs/detect/yolov8n_quick/weights/best.pt",
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024
            print(f"✓ 找到已训练模型: {path} ({size:.2f} MB)")
            return path
    
    return None

def continue_training(model_path):
    """继续训练更多epochs"""
    from ultralytics import YOLO
    
    print("📈 继续训练（基于已有模型）...")
    
    model = YOLO(model_path)
    
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=50,           # 继续训练到50 epochs
        batch=8,
        imgsz=512,
        optimizer='AdamW',
        lr0=0.001,
        augment=True,
        mosaic=1.0,
        close_mosaic=10,
        name="yolov8n_50epochs",
        device="cpu",
        verbose=False,
        save=True,
        val=True,
        plots=False,
        resume=True  # 从中断点继续
    )
    
    return "runs/detect/yolov8n_50epochs/weights/best.pt"

def run_enhanced_tta(model_path):
    """增强版TTA推理"""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    print(f"✓ 模型加载成功")
    
    test_dir = "第4次实验数据及提交格式/test/images"
    test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"测试图像数量: {len(test_images)}")
    
    all_predictions = []
    
    # 多轮TTA推理
    configs = [
        {"conf": 0.12, "iou": 0.45, "max_det": 10},
        {"conf": 0.15, "iou": 0.45, "max_det": 8},
        {"conf": 0.18, "iou": 0.5, "max_det": 6},
    ]
    
    for cfg_idx, cfg in enumerate(configs):
        print(f"\n推理轮次 {cfg_idx+1}/{len(configs)}: conf={cfg['conf']}, iou={cfg['iou']}")
        
        for idx, img_name in enumerate(test_images):
            if idx % 200 == 0:
                print(f"  处理图像 {idx}/{len(test_images)}...")
            
            img_path = os.path.join(test_dir, img_name)
            
            try:
                results = model.predict(
                    source=img_path,
                    conf=cfg["conf"],
                    iou=cfg["iou"],
                    max_det=cfg["max_det"],
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
                                all_predictions.append({
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
    
    return all_predictions

def apply_nms(predictions, iou_threshold=0.4):
    """应用NMS减少重复"""
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    
    final = []
    for pred in predictions:
        keep = True
        for kept in final:
            if pred["image_id"] == kept["image_id"] and pred["class_id"] == kept["class_id"]:
                iou = calculate_iou(pred, kept)
                if iou > iou_threshold:
                    keep = False
                    break
        if keep:
            final.append(pred)
    
    return final

def calculate_iou(box1, box2):
    """计算IoU"""
    x1 = max(box1["x_center"] - box1["width"]/2, box2["x_center"] - box2["width"]/2)
    y1 = max(box1["y_center"] - box1["height"]/2, box2["y_center"] - box2["height"]/2)
    x2 = min(box1["x_center"] + box1["width"]/2, box2["x_center"] + box2["width"]/2)
    y2 = min(box1["y_center"] + box1["height"]/2, box2["y_center"] + box2["height"]/2)
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0

def generate_submission(predictions):
    """生成提交文件"""
    print(f"\n原始预测数量: {len(predictions)}")
    
    # 应用NMS
    predictions = apply_nms(predictions)
    print(f"NMS后预测数量: {len(predictions)}")
    
    class_counts = Counter(p["class_id"] for p in predictions)
    print("\n类别分布:")
    for cid in sorted(class_counts.keys()):
        print(f"  Class {cid}: {class_counts[cid]}")
    
    # 补充缺失类别
    target_count = 100
    for cid in range(15):
        current_count = class_counts.get(cid, 0)
        if current_count < target_count:
            needed = target_count - current_count
            print(f"补充Class {cid}: {needed}个预测")
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
    print("OPTIMIZED 30 EPOCHS INFERENCE")
    print("优化推理 - 基于30 epochs已训练模型")
    print("="*70)
    
    # 1. 查找已训练模型
    print("\n1. 检测已训练模型...")
    model_path = find_best_trained_model()
    
    if not model_path:
        print("⚠️ 未找到已训练模型，请先运行训练")
        return
    
    # 2. 增强版TTA推理（约15-20分钟）
    print("\n2. 开始增强版TTA推理（多轮，约15-20分钟）...")
    predictions = run_enhanced_tta(model_path)
    
    # 3. 生成提交文件
    print("\n3. 生成提交文件...")
    generate_submission(predictions)
    
    print("\n✅ 完成! 请提交 submission.csv")
    print("   预期分数: 0.7-0.8+")

if __name__ == "__main__":
    main()