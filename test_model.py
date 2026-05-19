# -*- coding: utf-8 -*-
"""
测试YOLO模型是否能正常推理
"""

import os

print("="*70)
print("TEST YOLO MODEL")
print("测试YOLO模型是否能正常推理")
print("="*70)

# 尝试导入YOLO
try:
    from ultralytics import YOLO
    print("✓ 成功导入YOLO")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    exit()

# 加载模型
model_path = "runs/detect/traffic_signs_complete/weights/best.pt"
print(f"\n加载模型: {model_path}")

try:
    model = YOLO(model_path)
    print("✓ 模型加载成功")
    
    # 打印模型信息
    print("\n模型信息:")
    print(f"  类型: {type(model)}")
    print(f"  类别数量: {len(model.names)}")
    
    # 测试推理
    test_img = "第4次实验数据及提交格式/test/images/000000.jpg"
    print(f"\n测试推理: {test_img}")
    
    if os.path.exists(test_img):
        results = model.predict(test_img, conf=0.1, verbose=True)
        print("\n推理结果:")
        for result in results:
            if hasattr(result, 'boxes'):
                print(f"  检测到 {len(result.boxes)} 个目标")
                for box in result.boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    print(f"    类别: {cls}, 置信度: {conf:.4f}")
    else:
        print(f"✗ 测试图像不存在")
        
except Exception as e:
    print(f"✗ 模型加载或推理失败: {e}")
    import traceback
    traceback.print_exc()