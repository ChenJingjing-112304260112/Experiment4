"""
检查训练数据统计信息
"""

import os
from collections import Counter

def check_data():
    label_dir = "第4次实验数据及提交格式/train/labels"
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
    
    print("="*60)
    print("Training Data Analysis")
    print("="*60)
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    print(f"Total label files: {len(label_files)}")
    
    class_counts = Counter()
    total_objects = 0
    
    for label_file in label_files:
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total_objects += 1
    
    print(f"\nTotal objects: {total_objects}")
    print("\nClass distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_objects) * 100
        print(f"  {class_id:2d} - {class_names.get(class_id, 'Unknown'):20s}: {count:4d} ({percentage:5.2f}%)")
    
    # 检查是否有缺失的类别
    missing_classes = [cid for cid in range(15) if cid not in class_counts]
    if missing_classes:
        print(f"\nMissing classes: {missing_classes}")
    else:
        print("\nAll 15 classes present!")

if __name__ == "__main__":
    check_data()