import json

with open('traffic-signs-detection-using-yolov8.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 查找所有代码cell
for i, cell in enumerate(data['cells']):
    if cell.get('cell_type') != 'code':
        continue
    if 'source' not in cell:
        continue
    source = ''.join(cell['source'])
    
    # 查找包含文件操作、提交相关的代码
    if any(keyword in source.lower() for keyword in ['submission', 'sample', 'test.csv', 'detect', 'output', 'result']):
        print(f"{'='*60}")
        print(f"CELL {i}")
        print(f"{'='*60}")
        print(source)
        print("\n\n")