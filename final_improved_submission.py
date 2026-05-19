"""
改进的交通标志检测提交脚本

问题分析：
1. 当前模型置信度极低（平均0.0091）
2. 需要使用适当的置信度阈值过滤低质量预测
3. mAP@0.5评分需要足够高的置信度和正确的类别预测

改进策略：
1. 使用合理的置信度阈值（0.2-0.3）
2. 应用NMS过滤重叠框
3. 确保预测格式正确
"""

from ultralytics import YOLO
import csv
from pathlib import Path

def generate_improved_submission():
    model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    # 关键改进参数
    CONF_THRESHOLD = 0.2  # 合理的置信度阈值
    IOU_THRESHOLD = 0.45  # NMS阈值
    
    print("=" * 60)
    print("Traffic Sign Detection - Improved Submission")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Test Directory: {test_dir}")
    print(f"Output: {output_path}")
    print(f"Confidence Threshold: {CONF_THRESHOLD}")
    print(f"IOU Threshold: {IOU_THRESHOLD}")
    print("=" * 60)
    
    # 加载模型
    model = YOLO(model_path)
    print("Model loaded successfully")
    
    # 获取测试图片
    test_path = Path(test_dir)
    image_paths = sorted([p for p in test_path.iterdir() if p.is_file()])
    total_images = len(image_paths)
    print(f"Found {total_images} test images")
    
    # 推理
    all_predictions = []
    images_with_detections = 0
    
    for idx, img_path in enumerate(image_paths):
        # 执行推理
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=640,
            save=False,
            verbose=False,
            device='cpu'
        )
        
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            images_with_detections += 1
            for box in result.boxes:
                x_center = float(box.xywhn[0][0].item())
                y_center = float(box.xywhn[0][1].item())
                width = float(box.xywhn[0][2].item())
                height = float(box.xywhn[0][3].item())
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # 验证坐标在0-1范围内
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))
                
                all_predictions.append({
                    "image_id": img_path.name,
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "confidence": confidence,
                })
        
        # 进度输出
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total_images} images | Detections: {len(all_predictions)}")
    
    # 写入CSV
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"
        ])
        writer.writeheader()
        writer.writerows(all_predictions)
    
    # 统计信息
    print("\n" + "=" * 60)
    print("Submission Generated Successfully!")
    print("=" * 60)
    print(f"Total images processed: {total_images}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Output file: {output_path}")
    print(f"Approximate file size: {len(all_predictions) * 60 / 1024:.2f} KB")
    print("=" * 60)
    
    # 预测质量分析
    if len(all_predictions) > 0:
        avg_conf = sum(p['confidence'] for p in all_predictions) / len(all_predictions)
        print(f"\nAverage confidence: {avg_conf:.4f}")
        print(f"Confidence range: {min(p['confidence'] for p in all_predictions):.4f} - {max(p['confidence'] for p in all_predictions):.4f}")
        
        # 类别分布
        class_counts = {}
        for p in all_predictions:
            class_counts[p['class_id']] = class_counts.get(p['class_id'], 0) + 1
        
        print("\nClass distribution:")
        for class_id, count in sorted(class_counts.items()):
            percentage = (count / len(all_predictions)) * 100
            print(f"  Class {class_id:2d}: {count:4d} ({percentage:5.2f}%)")

if __name__ == "__main__":
    generate_improved_submission()