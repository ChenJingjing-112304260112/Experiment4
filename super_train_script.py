
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

from ultralytics import YOLO
import time

print("Loading YOLOv8x model...")
model = YOLO("yolov8x.pt")

print("Starting training with 500 epochs...")
start_time = time.time()

results = model.train(
    data="第4次实验数据及提交格式/data.yaml",
    epochs=500,
    batch=2,
    imgsz=640,
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.0001,
    cos_lr=True,
    patience=100,
    augment=True,
    mosaic=1.0,
    mixup=0.5,
    copy_paste=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=45.0,
    translate=0.2,
    scale=0.5,
    shear=10.0,
    flipud=0.0,
    fliplr=0.5,
    name="super_training",
    device="cpu",
    verbose=True,
    save_period=25,
    val=True,
    plots=True,
    seed=42,
    dropout=0.1
)

elapsed = time.time() - start_time
print(f"Training completed in {elapsed/3600:.2f} hours")
print(f"Final mAP@0.5: {results.box.map50:.4f}")

# 保存结果
with open("super_training_results.txt", "w") as f:
    f.write(f"mAP@0.5: {results.box.map50:.4f}\\n")
    f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\\n")
    f.write(f"Epochs: {results.epoch}\\n")
    f.write(f"Time: {elapsed/3600:.2f} hours\\n")
