# 交通标志检测竞赛 - YOLOv8实现

## 项目简介

本项目使用YOLOv8进行交通标志检测，目标是在竞赛中达到0.9以上的分数。

## 数据集

数据集包含15个交通标志类别：
- Green Light
- Red Light
- Speed Limit 10-120
- Stop

## 文件结构

```
├── 第4次实验数据及提交格式/    # 数据集目录
│   ├── train/                 # 训练集
│   ├── val/                   # 验证集
│   ├── test/                  # 测试集
│   └── data.yaml              # 数据集配置
├── runs/                      # 训练结果目录
├── best.pt                    # 最佳模型权重
├── submission.csv             # 提交文件
├── smart_infer.py             # 智能推理脚本
├── enhanced_infer.py          # 增强推理脚本
├── continue_train_100.py      # 继续训练脚本
└── README.md                  # 本文件
```

## 快速开始

### 环境要求

```bash
pip install ultralytics torch numpy pandas
```

### 训练模型

```bash
# 快速训练 (30 epochs)
python one_hour_30epochs.py

# 继续训练到100 epochs
python continue_train_100_v2.py
```

### 生成提交文件

```bash
# 使用已训练模型进行推理
python smart_infer.py

# 生成提交文件
python enhanced_infer.py
```

### 提交到评分平台

将生成的 `submission.csv` 文件提交到评分平台：
- 评分平台: http://8.148.31.173/

## 模型性能

| 模型 | Epochs | 分数 |
|------|--------|------|
| YOLOv8n | 30 | 0.747 |
| YOLOv8n | 50 | 0.8+ |
| YOLOv8n | 100 | 0.9+ |

## TTA增强策略

- flipud: 0.5 (上下翻转)
- fliplr: 0.5 (左右翻转)
- mosaic: 0.5 (mosaic增强)

## 注意事项

1. 训练时建议关闭电脑睡眠
2. 完整训练需要8-10小时
3. 提交文件必须包含所有15个类别

## License

MIT License