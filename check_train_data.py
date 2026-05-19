"""
检查训练数据中的类别分布
"""

import os
from collections import Counter

label_dir = "第4次实验数据及提交格式/train/labels"
class_counts = Counter()

for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

print("训练数据类别分布:")
for cid in range(15):
    count = class_counts.get(cid, 0)
    percentage = (count / sum(class_counts.values())) * 100 if class_counts else 0
    print(f"  Class {cid}: {count} ({percentage:.2f}%)")

# 检查验证集
val_label_dir = "第4次实验数据及提交格式/val/labels"
val_counts = Counter()

for label_file in os.listdir(val_label_dir):
    if label_file.endswith('.txt'):
        with open(os.path.join(val_label_dir, label_file), 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    val_counts[class_id] += 1

print("\n验证数据类别分布:")
for cid in range(15):
    count = val_counts.get(cid, 0)
    percentage = (count / sum(val_counts.values())) * 100 if val_counts else 0
    print(f"  Class {cid}: {count} ({percentage:.2f}%)")