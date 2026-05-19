"""
验证提交文件是否覆盖所有15个类别
"""

import csv
from collections import Counter

def main():
    print("="*70)
    print("VERIFYING SUBMISSION FILE")
    print("="*70)
    
    # 读取提交文件
    predictions = []
    with open("submission.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append({
                "image_id": row['image_id'],
                "class_id": int(row['class_id']),
                "confidence": float(row['confidence'])
            })
    
    print("Total predictions:", len(predictions))
    
    # 统计类别分布
    class_counts = Counter(p['class_id'] for p in predictions)
    
    print("\nClass distribution:")
    print("-" * 50)
    
    # 按类别ID排序输出
    total_predictions = len(predictions)
    all_classes_present = True
    missing_classes = []
    
    for class_id in range(15):
        count = class_counts.get(class_id, 0)
        percentage = (count / total_predictions) * 100 if total_predictions > 0 else 0
        
        if count == 0:
            status = "MISSING"
            all_classes_present = False
            missing_classes.append(class_id)
        else:
            status = "OK"
        
        print("Class %2d: %5d predictions (%5.2f%%) %s" % (class_id, count, percentage, status))
    
    # 检查结果
    print("\n" + "="*70)
    if all_classes_present:
        print("SUCCESS: All 15 classes are covered!")
    else:
        print("FAILURE: Missing classes:", missing_classes)
    
    # 统计置信度信息
    confidences = [p['confidence'] for p in predictions]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    min_conf = min(confidences) if confidences else 0
    max_conf = max(confidences) if confidences else 0
    
    print("\nConfidence statistics:")
    print("  Average confidence: %.4f" % avg_conf)
    print("  Minimum confidence: %.4f" % min_conf)
    print("  Maximum confidence: %.4f" % max_conf)
    
    # 检查图像覆盖情况
    unique_images = len(set(p['image_id'] for p in predictions))
    print("\nImage coverage:")
    print("  Unique images with predictions:", unique_images)

if __name__ == "__main__":
    main()