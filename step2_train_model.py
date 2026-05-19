"""
分步训练脚本 - Step 2: 训练模型（测试版本）
"""

from ultralytics import YOLO
import time

print("="*70)
print("STEP 2: TRAINING MODEL (10 epochs test)")
print("="*70)
print("Configuration:")
print("  - Model: YOLOv8m (medium)")
print("  - Epochs: 10")
print("  - Batch size: 8")
print("  - Data: combined_dataset")
print("  - Optimizer: AdamW")
print("  - Augmentation: enabled")
print("="*70)

start_time = time.time()

# 训练模型
model = YOLO("yolov8m.pt")

results = model.train(
    data="combined_data.yaml",
    epochs=10,
    batch=8,
    imgsz=640,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    augment=True,
    mosaic=1.0,
    mixup=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    flipud=0.0,
    fliplr=0.5,
    name="traffic_signs_optimized",
    device="cpu",
    verbose=True
)

elapsed_time = time.time() - start_time

print("\n" + "="*70)
print("TRAINING COMPLETED!")
print("="*70)
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(f"Best mAP@0.5: {results.box.map50:.4f}")
print(f"Final mAP@0.5:0.95: {results.box.map:.4f}")

# 保存结果
with open("training_results.txt", "w") as f:
    f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")
    f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")

print("\nStep 2 completed!")