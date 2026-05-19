"""
查看提交文件内容
"""

import csv

with open("submission.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    print("Header:", header)
    
    # 查看前10行数据
    print("\nFirst 10 rows:")
    for i, row in enumerate(reader):
        if i >= 10:
            break
        print(row)