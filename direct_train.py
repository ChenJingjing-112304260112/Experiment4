# -*- coding: utf-8 -*-
"""
使用YOLO Python API直接训练
"""

import os
os.environ['YOLO_VERBOSE'] = 'True'

print("="*70)
print("YOLOv8 DIRECT TRAINING")
print("="*70)

try:
    from ultralytics import YOLO
    print("✓ Ultralytics导入成功")

    # 加载模型
    model = YOLO("yolov8s.pt")
    print("✓ 模型加载成功")

    # 训练
    print("\n开始训练 (100 epochs)...")
    print("这可能需要几分钟，请等待...\n")

    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=100,
        batch=4,
        imgsz=640,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        augment=True,
        mosaic=1.0,
        mixup=0.2,
        name="yolov8_direct_train",
        device="cpu",
        verbose=True,
        save=True,
        val=True,
        plots=True
    )

    print("\n训练完成!")
    print(f"结果: {results}")

except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()