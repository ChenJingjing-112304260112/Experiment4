import json

with open('traffic-signs-detection-using-yolov8.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total cells: {len(data['cells'])}\n")

# 查找包含提交相关代码的cell
for i, cell in enumerate(data['cells']):
    if 'source' not in cell:
        continue
    source = ''.join(cell['source'])
    
    # 查找包含sample submission, test, predict等关键词的cell
    if any(keyword in source.lower() for keyword in ['sample', 'submission', 'test', 'predict', 'csv']):
        print(f"{'='*60}")
        print(f"CELL {i} - {cell.get('cell_type', 'unknown')}")
        print(f"{'='*60}")
        print(source)
        print("\n")