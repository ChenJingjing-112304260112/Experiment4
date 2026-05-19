"""
深度分析问题并提供改进方案
"""

import os
from collections import Counter

def analyze_data():
    print("="*70)
    print("DATA ANALYSIS")
    print("="*70)
    
    # 检查训练数据
    label_dir = "第4次实验数据及提交格式/train/labels"
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    print(f"Total training images: {len(label_files)}")
    
    class_counts = Counter()
    for label_file in label_files:
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_counts[int(parts[0])] += 1
    
    print("\nTraining class distribution:")
    total = sum(class_counts.values())
    for cid in sorted(class_counts.keys()):
        print(f"  Class {cid}: {class_counts[cid]} ({class_counts[cid]/total*100:.2f}%)")
    
    # 检查验证数据
    val_dir = "第4次实验数据及提交格式/val/labels"
    val_files = [f for f in os.listdir(val_dir) if f.endswith('.txt')]
    print(f"\nTotal validation images: {len(val_files)}")
    
    # 检查test数据
    test_dir = "第4次实验数据及提交格式/test/images"
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    print(f"Total test images: {len(test_files)}")

def analyze_model():
    print("\n" + "="*70)
    print("MODEL ANALYSIS")
    print("="*70)
    
    from ultralytics import YOLO
    
    # 检查现有模型
    model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"Model loaded: {model_path}")
        print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    else:
        print(f"Model not found: {model_path}")
    
    # 检查训练目录
    runs_dir = "runs/detect"
    if os.path.exists(runs_dir):
        experiments = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        print(f"\nTraining experiments found: {len(experiments)}")
        for exp in sorted(experiments):
            exp_path = os.path.join(runs_dir, exp)
            weights_path = os.path.join(exp_path, "weights")
            if os.path.exists(weights_path):
                weights = [f for f in os.listdir(weights_path) if f.endswith('.pt')]
                if weights:
                    print(f"  {exp}: {weights}")

def suggest_improvements():
    print("\n" + "="*70)
    print("IMPROVEMENT SUGGESTIONS")
    print("="*70)
    
    print("""
1. INCREASE TRAINING EPOCHS
   - Current: 10-30 epochs
   - Recommend: 100-200 epochs
   - Reason: Model needs more time to converge

2. USE LARGER MODEL
   - Current: yolov8s.pt
   - Recommend: yolov8m.pt or yolov8l.pt
   - Reason: Larger models have more capacity for complex patterns

3. ADJUST HYPERPARAMETERS
   - Learning rate: 0.001 → 0.0005
   - Batch size: 4 → 2 (if memory limited)
   - Optimizer: AdamW (already using)

4. DATA AUGMENTATION
   - Enable mosaic, mixup, flip, rotation
   - Increase HSV augmentation

5. CONFIDENCE THRESHOLD
   - Current: 0.1-0.2
   - Recommend: 0.001-0.05 for more predictions
   - Reason: Competition may need more detections

6. TTA (Test-Time Augmentation)
   - Enable during inference for better results
""")

if __name__ == "__main__":
    analyze_data()
    analyze_model()
    suggest_improvements()