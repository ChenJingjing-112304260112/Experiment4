from ultralytics import YOLO
import csv
from pathlib import Path
import sys

def main():
    model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    # 改进参数
    CONF_THRESHOLD = 0.25  # 提高置信度阈值，过滤低质量预测
    IOU_THRESHOLD = 0.45   # NMS阈值
    
    print("="*60)
    print("Traffic Sign Detection - Improved Submission")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"IOU threshold: {IOU_THRESHOLD}")
    print()
    
    # 加载模型
    model = YOLO(model_path)
    
    # 获取测试图片
    test_path = Path(test_dir)
    image_paths = sorted([p for p in test_path.iterdir() if p.is_file()])
    print(f"Found {len(image_paths)} test images")
    
    # 推理（使用改进的参数）
    print("\nRunning inference with improved parameters...")
    all_predictions = []
    
    for idx, img_path in enumerate(image_paths):
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=640,
            save=False,
            verbose=False,
            device='cpu',
            agnostic_nms=False  # 类别感知NMS
        )
        
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                pred = {
                    "image_id": img_path.name,
                    "class_id": int(box.cls[0].item()),
                    "x_center": box.xywhn[0][0].item(),
                    "y_center": box.xywhn[0][1].item(),
                    "width": box.xywhn[0][2].item(),
                    "height": box.xywhn[0][3].item(),
                    "confidence": box.conf[0].item(),
                }
                all_predictions.append(pred)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)} images, {len(all_predictions)} predictions")
    
    # 写入CSV
    print(f"\nWriting {len(all_predictions)} predictions to {output_path}...")
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(all_predictions)
    
    print(f"\n{'='*60}")
    print(f"Improved submission created!")
    print(f"  Total predictions: {len(all_predictions)}")
    print(f"  Output file: {output_path}")
    print(f"  Confidence threshold used: {CONF_THRESHOLD}")
    print("="*60)

if __name__ == "__main__":
    main()