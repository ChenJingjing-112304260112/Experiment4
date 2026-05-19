import os

label_dir = '第4次实验数据及提交格式/train/labels'
labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

counts = [0] * 15
for label_file in labels:
    with open(os.path.join(label_dir, label_file), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cid = int(parts[0])
                if 0 <= cid < 15:
                    counts[cid] += 1

total = sum(counts)
print('Class distribution:')
for i in range(15):
    print(f'Class {i}: {counts[i]} ({counts[i]/total*100:.2f}%)')
print(f'Total: {total}')