import os

weights_dir = "runs/detect/yolov8s_final/weights"
if os.path.exists(weights_dir):
    weights = os.listdir(weights_dir)
    print(f"权重文件: {weights}")
    for w in weights:
        size = os.path.getsize(os.path.join(weights_dir, w)) / 1024 / 1024
        print(f"{w}: {size:.2f} MB")
else:
    print("训练目录不存在")