"""
最终尝试 - 使用所有数据训练并优化提交
"""

from ultralytics import YOLO
import csv
from pathlib import Path
import subprocess
import shutil

def combine_datasets():
    """合并训练集和验证集"""
    print("Combining train and validation datasets...")
    
    # 创建新的数据目录
    combined_dir = "combined_dataset"
    os.makedirs(os.path.join(combined_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(combined_dir, "labels"), exist_ok=True)
    
    # 复制训练数据
    train_images = "第4次实验数据及提交格式/train/images"
    train_labels = "第4次实验数据及提交格式/train/labels"
    for f in os.listdir(train_images):
        if f.endswith('.jpg'):
            shutil.copy(os.path.join(train_images, f), os.path.join(combined_dir, "images", f))
            label_file = f.replace('.jpg', '.txt')
            if os.path.exists(os.path.join(train_labels, label_file)):
                shutil.copy(os.path.join(train_labels, label_file), os.path.join(combined_dir, "labels", label_file))
    
    # 复制验证数据（重命名避免冲突）
    val_images = "第4次实验数据及提交格式/val/images"
    val_labels = "第4次实验数据及提交格式/val/labels"
    idx = 0
    for f in os.listdir(val_images):
        if f.endswith('.jpg'):
            new_name = f"val_{idx}_{f}"
            shutil.copy(os.path.join(val_images, f), os.path.join(combined_dir, "images", new_name))
            label_file = f.replace('.jpg', '.txt')
            if os.path.exists(os.path.join(val_labels, label_file)):
                shutil.copy(os.path.join(val_labels, label_file), os.path.join(combined_dir, "labels", new_name.replace('.jpg', '.txt')))
            idx += 1
    
    # 创建新的data.yaml
    data_yaml = """
train: ../combined_dataset/images
val: ../第4次实验数据及提交格式/val/images
test: ../第4次实验数据及提交格式/test/images

nc: 15
names: ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110', 
       'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50', 
       'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop']
"""
    with open("combined_data.yaml", 'w') as f:
        f.write(data_yaml)
    
    print(f"Combined dataset created with {len(os.listdir(os.path.join(combined_dir, 'images')))} images")

def train_and_infer():
    """训练并生成提交"""
    import os
    
    # 合并数据集
    combine_datasets()
    
    # 训练模型
    print("\nTraining with combined dataset...")
    model = YOLO('yolov8s.pt')
    
    results = model.train(
        data='combined_data.yaml',
        epochs=100,
        batch=2,
        imgsz=640,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        augment=True,
        name='traffic_signs_combined',
        verbose=True,
        device='cpu'
    )
    
    # 验证
    metrics = model.val()
    mAP50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
    print(f"\nValidation mAP50: {mAP50:.4f}")
    
    # 生成提交
    test_dir = "第4次实验数据及提交格式/test/images"
    output_path = "submission.csv"
    
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.is_file()])
    predictions = []
    
    # 使用非常低的置信度阈值
    for img_path in image_paths:
        results = model.predict(source=str(img_path), conf=0.001, verbose=False)
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
    
    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
    
    # 上传到GitHub
    subprocess.run(["git", "add", "submission.csv", "final_attempt.py"])
    subprocess.run(["git", "commit", "-m", f"Final attempt: {len(predictions)} preds, mAP50={mAP50:.4f}"])
    subprocess.run(["git", "push", "origin", "main"])
    
    print(f"\nGenerated {len(predictions)} predictions")
    print(f"Average confidence: {avg_conf:.4f}")

if __name__ == "__main__":
    import os
    train_and_infer()