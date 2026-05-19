"""
评估模型在验证集上的性能
分析为什么分数低
"""

from ultralytics import YOLO

def main():
    print("="*70)
    print("EVALUATING MODEL PERFORMANCE")
    print("="*70)
    
    # 加载模型
    model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")
    
    # 在验证集上评估
    print("Running validation...")
    results = model.val(data="第4次实验数据及提交格式/data.yaml", verbose=True)
    
    # 打印关键指标
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print("mAP@0.5: %.4f" % results.box.map50)
    print("mAP@0.5:0.95: %.4f" % results.box.map)
    
    # 打印每个类别的AP
    print("\nPer-class AP@0.5:")
    print("-" * 50)
    for i, ap in enumerate(results.box.ap50):
        print("Class %2d: AP@0.5 = %.4f" % (i, ap))
    
    # 检查哪些类别性能差
    print("\nClasses with low AP (< 0.3):")
    low_ap_classes = []
    for i, ap in enumerate(results.box.ap50):
        if ap < 0.3:
            print("Class %2d: AP@0.5 = %.4f (LOW)" % (i, ap))
            low_ap_classes.append(i)
    
    if low_ap_classes:
        print("\nWARNING: These classes have very low performance!")
        print("Low AP classes:", low_ap_classes)
    else:
        print("\nAll classes have acceptable performance.")

if __name__ == "__main__":
    main()