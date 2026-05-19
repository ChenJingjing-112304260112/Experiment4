# 环境诊断脚本
import sys
print("Python版本:", sys.version)

# 检查ultralytics安装
try:
    import ultralytics
    print("Ultralytics版本:", ultralytics.__version__)
except ImportError as e:
    print("Ultralytics未安装:", e)
    
# 检查torch
try:
    import torch
    print("PyTorch版本:", torch.__version__)
    print("CUDA可用:", torch.cuda.is_available())
except ImportError as e:
    print("PyTorch未安装:", e)

# 检查工作目录
import os
print("工作目录:", os.getcwd())
print("balanced_data存在:", os.path.exists("balanced_data"))

# 尝试简单的YOLO导入
try:
    from ultralytics import YOLO
    print("YOLO导入成功")
except Exception as e:
    print("YOLO导入失败:", e)