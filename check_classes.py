import os
from collections import Counter

label_dir = '第4次实验数据及提交格式/train/labels'
labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

class_counts = Counter()
for label_file in labels:
    with open(os.path.join(label_dir, label_file), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_counts[int(parts[0])] += 1

print('训练集类别分布:')
total = sum(class_counts.values())
for cid in sorted(class_counts.keys()):
    print(f'  Class {cid:2d}: {class_counts[cid]:4d} ({class_counts[cid]/total*100:.2f}%)')
print(f'总标注数: {total}')