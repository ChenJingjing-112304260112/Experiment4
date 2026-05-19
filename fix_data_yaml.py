# 更新data.yaml使用绝对路径
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "traffic_signs_data")

data_yaml = f"""path: {data_dir}
train: train/images
val: val/images
test: test/images
nc: 15
names:
  0: Green Light
  1: Red Light
  2: Speed Limit 10
  3: Speed Limit 100
  4: Speed Limit 110
  5: Speed Limit 120
  6: Speed Limit 20
  7: Speed Limit 30
  8: Speed Limit 40
  9: Speed Limit 50
  10: Speed Limit 60
  11: Speed Limit 70
  12: Speed Limit 80
  13: Speed Limit 90
  14: Stop
"""

with open(os.path.join(data_dir, "data.yaml"), "w") as f:
    f.write(data_yaml)

print("data.yaml已更新，使用绝对路径:", data_dir)