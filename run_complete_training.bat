@echo off
chcp 65001 >nul
title YOLOv8 交通标志检测 - 完整训练脚本

echo ============================================================
echo YOLOv8 交通标志检测 - 完整本地训练脚本
echo 目标: mAP@0.5 >= 0.9
echo ============================================================
echo.

cd /d %~dp0

echo [步骤 1] 检查Python环境...
python --version
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo.
echo [步骤 2] 安装必要的包...
pip install ultralytics>=8.0.0 torch>=2.0.0 opencv-python pandas numpy PyYAML -q

echo.
echo [步骤 3] 检查数据目录...
if not exist "第4次实验数据及提交格式" (
    echo [错误] 数据目录不存在!
    pause
    exit /b 1
)

echo.
echo [步骤 4] 开始模型训练...
echo ============================================================
echo 训练配置:
echo   模型: YOLOv8s
echo   训练轮数: 100 epochs
echo   批量大小: 8
echo   图像尺寸: 640x640
echo ============================================================
echo.

python -m ultralytics train ^
    data="第4次实验数据及提交格式/data.yaml" ^
    model=yolov8s.pt ^
    epochs=100 ^
    batch=8 ^
    imgsz=640 ^
    optimizer=AdamW ^
    lr0=0.001 ^
    cos_lr=True ^
    augment=True ^
    name=traffic_signs_final ^
    device=0 ^
    verbose=True ^
    save=True ^
    val=True

if errorlevel 1 (
    echo [警告] GPU训练失败，尝试CPU...
    python -m ultralytics train ^
        data="第4次实验数据及提交格式/data.yaml" ^
        model=yolov8s.pt ^
        epochs=100 ^
        batch=4 ^
        imgsz=640 ^
        optimizer=AdamW ^
        augment=True ^
        name=traffic_signs_final ^
        device=cpu ^
        verbose=True
)

echo.
echo [步骤 5] 生成提交文件...
python generate_submission.py

echo.
echo ============================================================
echo 训练完成!
echo ============================================================
echo.
echo 提交文件: submission.csv
echo.
echo 请将submission.csv提交到评分平台
echo.

pause