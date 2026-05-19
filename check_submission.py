import csv

with open('submission.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f'预测数量: {len(rows)}')

class_counts = {}
for row in rows:
    cid = int(row['class_id'])
    class_counts[cid] = class_counts.get(cid, 0) + 1

print('\n类别分布:')
for cid in sorted(class_counts.keys()):
    print(f'  Class {cid}: {class_counts[cid]}')

print('\n前5行:')
for row in rows[:5]:
    print(f'  {row}')