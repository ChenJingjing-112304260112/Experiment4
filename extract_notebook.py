import json

with open('traffic-signs-detection-using-yolov8.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total cells: {len(data['cells'])}\n")

for i, cell in enumerate(data['cells']):
    if 'source' in cell:
        source = ''.join(cell['source'])
        if 'sample' in source.lower() or 'submission' in source.lower() or 'test' in source.lower():
            print(f"=== Cell {i} ===")
            print(source[:2000])
            print("\n" + "="*50 + "\n")