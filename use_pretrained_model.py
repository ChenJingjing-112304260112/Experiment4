"""
使用预训练的YOLOv8模型进行推理
不使用我们训练的模型，直接使用官方预训练模型
"""

from ultralytics import YOLO
import csv
from pathlib import Path

def main():
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    print("="*60)
    print("Using Pretrained YOLOv8 Model")
    print("="*60)
    
    # 使用官方预训练模型
    print("Loading YOLOv8n pretrained model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded")
    
    test_path = Path(test_dir)
    image_paths = sorted([p for p in test_path.iterdir() if p.is_file()])
    print(f"Found {len(image_paths)} test images")
    
    # YOLOv8预训练模型的类别列表
    # 我们需要映射交通标志类别到COCO类别
    coco_classes = {
        9: "stop sign",    # COCO类别9是stop sign
        10: "traffic light" # COCO类别10是traffic light
    }
    
    # 我们的类别映射
    our_classes = {
        0: "Green Light",
        1: "Red Light",
        14: "Stop"
    }
    
    all_predictions = []
    
    for idx, img_path in enumerate(image_paths):
        results = model.predict(
            source=str(img_path),
            conf=0.5,
            iou=0.45,
            imgsz=640,
            save=False,
            verbose=False,
            device='cpu'
        )
        
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                coco_class_id = int(box.cls[0].item())
                
                # 映射COCO类别到我们的类别
                if coco_class_id == 9:  # stop sign
                    our_class_id = 14
                elif coco_class_id == 10:  # traffic light
                    # 预训练模型不能区分红绿灯，我们默认预测为红灯
                    our_class_id = 1
                else:
                    continue  # 忽略其他类别
                
                all_predictions.append({
                    "image_id": img_path.name,
                    "class_id": our_class_id,
                    "x_center": float(box.xywhn[0][0].item()),
                    "y_center": float(box.xywhn[0][1].item()),
                    "width": float(box.xywhn[0][2].item()),
                    "height": float(box.xywhn[0][3].item()),
                    "confidence": float(box.conf[0].item()),
                })
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images")
    
    print(f"\nTotal predictions: {len(all_predictions)}")
    
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(all_predictions)
    
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    main()