# -*- coding: utf-8 -*-
"""
诊断和优化脚本
"""

import os
import sys

print("Python version:", sys.version)
print("Working directory:", os.getcwd())

# 检查数据集
data_dir = "第4次实验数据及提交格式"
print("\nData directory exists:", os.path.exists(data_dir))

# 检查数据配置
yaml_path = os.path.join(data_dir, "data.yaml")
print("YAML config exists:", os.path.exists(yaml_path))

if os.path.exists(yaml_path):
    with open(yaml_path, 'r') as f:
        print("\nYAML content:")
        print(f.read())

# 检查测试图像
test_dir = os.path.join(data_dir, "test", "images")
print(f"\nTest images count: {len(os.listdir(test_dir))}")

# 尝试加载模型
try:
    from ultralytics import YOLO
    print("\nUltralytics imported successfully")
    
    model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")
    print("Model loaded successfully")
    
    # 测试推理
    test_images = [f for f in os.listdir(test_dir) if f.endswith(".jpg")][:2]
    for img in test_images:
        results = model.predict(os.path.join(test_dir, img), conf=0.1, verbose=False)
        if results[0].boxes is not None:
            print(f"Image {img}: {len(results[0].boxes)} detections")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()