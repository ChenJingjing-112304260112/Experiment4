"""
简单的提交文件生成脚本
"""

from ultralytics import YOLO
import csv
import os

def main():
    model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    print("Loading model...")
    model = YOLO(model_path)
    
    print("Getting test images...")
    image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
    print(f"Found {len(image_files)} images")
    
    predictions = []
    
    for img_name in image_files:
        img_path = os.path.join(test_dir, img_name)
        results = model.predict(source=img_path, conf=0.1, verbose=False)
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                predictions.append({
                    "image_id": img_name,
                    "class_id": int(box.cls[0].item()),
                    "x_center": float(box.xywhn[0][0].item()),
                    "y_center": float(box.xywhn[0][1].item()),
                    "width": float(box.xywhn[0][2].item()),
                    "height": float(box.xywhn[0][3].item()),
                    "confidence": float(box.conf[0].item()),
                })
    
    print(f"Total predictions: {len(predictions)}")
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    main()