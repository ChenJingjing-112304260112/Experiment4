import os

# 检查模型目录
model_dirs = [
    "runs/detect/traffic_signs_complete",
    "runs/detect/yolov8s_full_train",
    "runs/detect/yolov8s_final"
]

for dir_path in model_dirs:
    if os.path.exists(dir_path):
        weights_dir = os.path.join(dir_path, "weights")
        if os.path.exists(weights_dir):
            weights = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
            if weights:
                print(f"找到模型目录: {dir_path}")
                for w in weights:
                    path = os.path.join(weights_dir, w)
                    size = os.path.getsize(path) / 1024 / 1024
                    print(f"  {w}: {size:.2f} MB")
            
            # 检查是否有训练日志
            results_file = os.path.join(dir_path, "results.csv")
            if os.path.exists(results_file):
                print(f"  找到训练日志")
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip()
                        print(f"  最后训练结果: {last_line[:50]}...")
        print()