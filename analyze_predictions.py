import csv
from collections import Counter

def analyze_submission(csv_path):
    """分析当前提交文件的预测分布"""
    
    print("="*60)
    print("Analyzing Submission Predictions")
    print("="*60)
    
    class_names = {
        0: "Green Light",
        1: "Red Light", 
        2: "Speed Limit 10",
        3: "Speed Limit 100",
        4: "Speed Limit 110",
        5: "Speed Limit 120",
        6: "Speed Limit 20",
        7: "Speed Limit 30",
        8: "Speed Limit 40",
        9: "Speed Limit 50",
        10: "Speed Limit 60",
        11: "Speed Limit 70",
        12: "Speed Limit 80",
        13: "Speed Limit 90",
        14: "Stop"
    }
    
    predictions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append({
                'class_id': int(row['class_id']),
                'confidence': float(row['confidence'])
            })
    
    print(f"\nTotal predictions: {len(predictions)}")
    
    # 类别分布
    class_counts = Counter(p['class_id'] for p in predictions)
    print("\nClass distribution:")
    for class_id, count in sorted(class_counts.items()):
        percentage = (count / len(predictions)) * 100
        print(f"  {class_id:2d} - {class_names.get(class_id, 'Unknown'):20s}: {count:6d} ({percentage:5.2f}%)")
    
    # 置信度分布
    conf_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    conf_counts = Counter()
    for p in predictions:
        for i in range(len(conf_bins)-1):
            if conf_bins[i] <= p['confidence'] < conf_bins[i+1]:
                conf_counts[f"{conf_bins[i]:.1f}-{conf_bins[i+1]:.1f}"] += 1
                break
    
    print("\nConfidence distribution:")
    for bin_range, count in sorted(conf_counts.items()):
        percentage = (count / len(predictions)) * 100
        print(f"  {bin_range:10s}: {count:6d} ({percentage:5.2f}%)")
    
    # 平均置信度
    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
    print(f"\nAverage confidence: {avg_conf:.4f}")
    
    # 分析问题
    print("\n" + "="*60)
    print("Analysis Results:")
    print("="*60)
    if avg_conf < 0.3:
        print("⚠️  WARNING: Average confidence is very low (< 0.3)")
        print("  This indicates the model is not confident in its predictions")
    
    if max(class_counts.values()) / len(predictions) > 0.3:
        print("⚠️  WARNING: One class dominates predictions")
        print("  This may indicate model bias or overfitting")
    
    print("\nSuggestions:")
    print("1. Increase confidence threshold to filter low-quality predictions")
    print("2. Check if model was properly trained on all classes")
    print("3. Consider using larger model (YOLOv8s/m/l)")
    print("4. Increase training epochs")

if __name__ == "__main__":
    analyze_submission("submission.csv")