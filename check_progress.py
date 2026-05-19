import os

# 检查训练进度
weights_path = "runs/detect/yolov8n_30epochs/weights"

if os.path.exists(weights_path):
    files = os.listdir(weights_path)
    print("训练权重文件:")
    for f in files:
        size = os.path.getsize(os.path.join(weights_path, f)) / 1024 / 1024
        print(f"  {f}: {size:.2f} MB")
    
    # 检查训练日志
    results_file = "runs/detect/yolov8n_30epochs/results.csv"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            lines = f.readlines()
            print(f"\n训练日志行数: {len(lines)-1} (已完成{len(lines)-2}个epoch)")
            if len(lines) > 1:
                print("最后一行:")
                print(lines[-1].strip()[:100])
else:
    print("训练目录不存在，尚未开始训练")