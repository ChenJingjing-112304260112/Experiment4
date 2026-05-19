from ultralytics import YOLO
import csv

# 简单测试
model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")
test_img = "第4次实验数据及提交格式/test/images/000003_jpg.rf.8511b9c219dbf9799a6d58900b15917d.jpg"

print(f"Testing image: {test_img}")
print("="*50)

# 测试不同置信度阈值
for conf in [0.001, 0.1, 0.25, 0.5, 0.75]:
    results = model.predict(source=test_img, conf=conf, verbose=False)
    boxes = results[0].boxes
    count = len(boxes) if boxes is not None else 0
    print(f"Conf={conf:.3f}: {count} detections")

print("\n" + "="*50)
print("Test complete!")