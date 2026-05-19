# -*- coding: utf-8 -*-
"""
自动查找并测试所有可用的测试图片
"""

import os

print("="*70)
print("FIND AND TEST ALL TEST IMAGES")
print("自动查找并测试所有可用的测试图片")
print("="*70)

# 查找测试图像
print("\n1. 查找测试图像...")
test_dirs = [
    "第4次实验数据及提交格式/test/images",
    "第4次实验数据及提交格式/test",
    "test/images",
    "test"
]

test_images = []
test_dir = None

for td in test_dirs:
    if os.path.exists(td):
        test_dir = td
        for f in os.listdir(td):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(f)
        break

if test_images:
    print(f"✓ 找到测试目录: {test_dir}")
    print(f"✓ 找到测试图像: {len(test_images)} 张")
    print(f"  示例图像: {test_images[:5]}")
else:
    print("✗ 未找到测试图像")
    exit()

# 测试YOLO模型
print("\n2. 测试YOLO模型...")
try:
    from ultralytics import YOLO
    print("✓ 成功导入YOLO")
except Exception as e:
    print(f"✗ 导入YOLO失败: {e}")
    exit()

# 查找模型
print("\n3. 查找训练好的模型...")
model_paths = [
    "runs/detect/traffic_signs_complete/weights/best.pt",
    "runs/detect/yolov8_full_train/weights/best.pt",
    "runs/detect/yolov8_direct_train/weights/best.pt"
]

model_path = None
for mp in model_paths:
    if os.path.exists(mp):
        model_path = mp
        break

if model_path:
    print(f"✓ 找到模型: {model_path}")
else:
    print("✗ 未找到训练好的模型")
    exit()

# 加载模型
print("\n4. 加载模型...")
model = YOLO(model_path)
print("✓ 模型加载成功")
print(f"  类别数量: {len(model.names)}")
print(f"  类别: {model.names}")

# 测试推理
print("\n5. 测试推理...")
success_count = 0
total_detections = 0

# 只测试前10张图像
for idx, img_name in enumerate(test_images[:10]):
    img_path = os.path.join(test_dir, img_name)
    
    try:
        results = model.predict(img_path, conf=0.05, verbose=False)
        
        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            num_boxes = len(results[0].boxes)
            total_detections += num_boxes
            success_count += 1
            print(f"  ✓ {img_name}: 检测到 {num_boxes} 个目标")
            
            # 打印详细结果
            for box in results[0].boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                print(f"    - 类别: {cls} ({model.names[cls]}), 置信度: {conf:.4f}")
        else:
            print(f"  ✗ {img_name}: 未检测到目标")
            
    except Exception as e:
        print(f"  ✗ {img_name}: 推理失败 - {e}")

print("\n" + "="*70)
print("测试完成!")
print(f"成功测试: {success_count}/{len(test_images[:10])}")
print(f"总检测目标数: {total_detections}")
print("="*70)