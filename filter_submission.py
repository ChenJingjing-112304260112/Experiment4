"""
过滤提交文件中的低置信度预测
"""

import csv

def filter_submission(input_path, output_path, min_confidence):
    """过滤低置信度预测"""
    
    print(f"Reading {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        predictions = list(reader)
    
    print(f"Total predictions: {len(predictions)}")
    
    # 过滤低置信度
    filtered = [p for p in predictions if float(p['confidence']) >= min_confidence]
    
    print(f"Filtered predictions (conf >= {min_confidence}): {len(filtered)}")
    
    # 写入新文件
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
        writer.writeheader()
        writer.writerows(filtered)
    
    print(f"Filtered submission saved to {output_path}")
    
    # 统计信息
    if filtered:
        avg_conf = sum(float(p['confidence']) for p in filtered) / len(filtered)
        print(f"Average confidence after filtering: {avg_conf:.4f}")

if __name__ == "__main__":
    filter_submission("submission.csv", "submission_filtered.csv", 0.05)