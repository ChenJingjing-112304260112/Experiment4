"""
修复数据配置文件
"""

import os

print("Fixing data configuration file...")

# 使用绝对路径
base_dir = os.getcwd()
combined_dir = os.path.join(base_dir, "combined_dataset")

# 创建正确的数据配置
data_yaml = f"""
train: {os.path.join(combined_dir, "images")}
val: {os.path.join(combined_dir, "images")}
nc: 15
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
"""

with open("combined_data.yaml", "w") as f:
    f.write(data_yaml)

print("Data configuration fixed!")
print(f"Content:\n{data_yaml}")