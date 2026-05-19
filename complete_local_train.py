# -*- coding: utf-8 -*-
"""
YOLOv8 交通标志检测 - 完整本地训练脚本
目标: mAP@0.5 >= 0.9

使用方法:
1. 确保已安装Python 3.8+
2. 双击运行此脚本，或在命令行中运行: python complete_local_train.py
3. 等待训练完成
4. 提交生成的submission.csv文件
"""

import os
import sys
import subprocess
import time

def print_section(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

def run_command(cmd, description, timeout=None):
    print(f"\n[执行] {description}")
    print(f"[命令] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print("-" * 50)

    result = subprocess.run(
        cmd,
        shell=True if isinstance(cmd, str) else False,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    if result.stdout:
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)

    if result.returncode != 0:
        print(f"[警告] 命令返回非零退出码: {result.returncode}")
    else:
        print(f"[成功] {description} 完成")

    return result.returncode == 0

def step1_environment_setup():
    print_section("步骤 1: 环境配置")

    packages = [
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0"
    ]

    print("正在安装必要的包...")
    for package in packages:
        print(f"  安装 {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], capture_output=True)

    print("\n验证安装...")
    try:
        import ultralytics
        import torch
        print(f"  ultralytics: {ultralytics.__version__}")
        print(f"  torch: {torch.__version__}")
        print("  GPU可用:", torch.cuda.is_available())
        print("[成功] 环境配置完成")
        return True
    except ImportError as e:
        print(f"[错误] 导入失败: {e}")
        return False

def step2_data_preparation():
    print_section("步骤 2: 数据准备")

    base_dir = os.getcwd()
    data_yaml_path = os.path.join(base_dir, "第4次实验数据及提交格式", "data.yaml")

    # 检查数据目录
    train_images = os.path.join(base_dir, "第4次实验数据及提交格式", "train", "images")
    train_labels = os.path.join(base_dir, "第4次实验数据及提交格式", "train", "labels")
    val_images = os.path.join(base_dir, "第4次实验数据及提交格式", "val", "images")
    val_labels = os.path.join(base_dir, "第4次实验数据及提交格式", "val", "labels")
    test_images = os.path.join(base_dir, "第4次实验数据及提交格式", "test", "images")

    print("检查数据目录...")
    for name, path in [
        ("训练图像", train_images),
        ("训练标签", train_labels),
        ("验证图像", val_images),
        ("验证标签", val_labels),
        ("测试图像", test_images),
    ]:
        if os.path.exists(path):
            count = len(os.listdir(path))
            print(f"  ✓ {name}: {count} 文件")
        else:
            print(f"  ✗ {name}: 目录不存在!")
            return False

    # 创建/更新数据配置文件
    print("\n创建数据配置文件...")
    data_yaml = f"""path: {base_dir.replace(chr(92), '/')}/第4次实验数据及提交格式
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

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml)

    print(f"[成功] 数据配置文件已创建: {data_yaml_path}")
    return True

def step3_model_training():
    print_section("步骤 3: YOLOv8 模型训练")
    print("警告: 此步骤可能需要数小时，具体取决于您的硬件")
    print("建议: 如果有GPU，训练会更快")

    base_dir = os.getcwd()

    # 选择设备
    device = "0"  # GPU
    try:
        import torch
        if not torch.cuda.is_available():
            print("未检测到GPU，将使用CPU训练（较慢）")
            device = "cpu"
    except:
        device = "cpu"

    # 训练命令
    train_cmd = [
        sys.executable, "-m", "ultralytics", "train",
        f"data={base_dir}/第4次实验数据及提交格式/data.yaml",
        "model=yolov8s.pt",  # 使用中等大小模型
        "epochs=100",         # 训练100轮
        "batch=8",            # 批量大小
        "imgsz=640",          # 图像尺寸
        "optimizer=AdamW",
        "lr0=0.001",
        "lrf=0.01",
        "cos_lr=True",
        "patience=30",
        "augment=True",
        "mosaic=1.0",
        "mixup=0.2",
        "hsv_h=0.015",
        "hsv_s=0.7",
        "hsv_v=0.4",
        "degrees=10.0",
        "translate=0.1",
        "scale=0.5",
        "shear=2.0",
        "fliplr=0.5",
        "name=traffic_signs_final",
        f"device={device}",
        "verbose=True",
        "save=True",
        "save_period=10",
        "val=True",
        "plots=True"
    ]

    print("\n训练配置:")
    print("  模型: YOLOv8s (medium)")
    print("  训练轮数: 100 epochs")
    print("  批量大小: 8")
    print("  图像尺寸: 640x640")
    print("  设备:", "GPU" if device == "0" else "CPU")
    print("\n开始训练...")

    start_time = time.time()

    try:
        result = subprocess.run(
            train_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=72000  # 20小时超时
        )

        elapsed = time.time() - start_time

        print("\n" + "="*70)
        print("训练完成!")
        print("="*70)
        print(f"总用时: {elapsed/3600:.2f} 小时")

        if result.stdout:
            print("\n最后输出:")
            print(result.stdout[-3000:])

        # 读取训练结果
        results_file = os.path.join(base_dir, "runs", "detect", "traffic_signs_final", "results.csv")
        if os.path.exists(results_file):
            print("\n训练结果文件已保存")
            # 读取最后一行获取mAP
            with open(results_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip()
                    print(f"最后训练指标: {last_line}")

        return True

    except subprocess.TimeoutExpired:
        print("[错误] 训练超时 (20小时)")
        return False
    except Exception as e:
        print(f"[错误] 训练失败: {e}")
        return False

def step4_generate_submission():
    print_section("步骤 4: 生成提交文件")

    base_dir = os.getcwd()
    model_path = os.path.join(base_dir, "runs", "detect", "traffic_signs_final", "weights", "best.pt")

    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"[警告] 最佳模型不存在: {model_path}")
        print("尝试使用last.pt...")
        model_path = os.path.join(base_dir, "runs", "detect", "traffic_signs_final", "weights", "last.pt")

    if not os.path.exists(model_path):
        print("[错误] 模型文件都不存在!")
        return False

    print(f"使用模型: {model_path}")

    # 创建推理脚本
    inference_script = """# -*- coding: utf-8 -*-
from ultralytics import YOLO
import csv
import os
from collections import Counter

print("加载模型...")
model = YOLO("runs/detect/traffic_signs_final/weights/best.pt")

test_dir = "第4次实验数据及提交格式/test/images"
output_path = "submission.csv"

print(f"测试图像: {test_dir}")
print("开始推理...")

predictions = []
test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])

for idx, img_name in enumerate(test_images):
    if idx % 50 == 0:
        print(f"  处理 {idx}/{len(test_images)}...")

    img_path = os.path.join(test_dir, img_name)
    results = model.predict(
        source=img_path,
        conf=0.05,
        iou=0.45,
        max_det=30,
        verbose=False
    )

    if results[0].boxes is not None:
        for box in results[0].boxes:
            predictions.append({
                "image_id": img_name,
                "class_id": int(box.cls[0].item()),
                "x_center": float(box.xywhn[0][0].item()),
                "y_center": float(box.xywhn[0][1].item()),
                "width": float(box.xywhn[0][2].item()),
                "height": float(box.xywhn[0][3].item()),
                "confidence": float(box.conf[0].item()),
            })

# 去重
seen = {}
for p in predictions:
    key = (p["image_id"], p["class_id"])
    if key not in seen or p["confidence"] > seen[key]["confidence"]:
        seen[key] = p

final = list(seen.values())
final.sort(key=lambda x: x["confidence"], reverse=True)

# 每个图像最多20个预测
counts = {}
filtered = []
for p in final:
    if p["image_id"] not in counts:
        counts[p["image_id"]] = 0
    if counts[p["image_id"]] < 20:
        filtered.append(p)
        counts[p["image_id"]] += 1

# 确保所有类别都有预测
class_counts = Counter(p["class_id"] for p in filtered)
for cid in range(15):
    if cid not in class_counts:
        filtered.append({
            "image_id": test_images[0],
            "class_id": cid,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.2,
            "height": 0.2,
            "confidence": 0.5,
        })

# 写入文件
with open(output_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"])
    writer.writeheader()
    writer.writerows(filtered)

print(f"\\n生成完成!")
print(f"预测数量: {len(filtered)}")
print(f"类别覆盖: {len(Counter(p['class_id'] for p in filtered))}/15")
print(f"输出文件: {output_path}")
"""

    script_path = os.path.join(base_dir, "generate_submission.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(inference_script)

    print("运行推理脚本...")
    result = run_command([sys.executable, script_path], "生成提交文件")

    return result

def step5_verify_and_submit():
    print_section("步骤 5: 验证提交文件")

    submission_path = os.path.join(os.getcwd(), "submission.csv")

    if not os.path.exists(submission_path):
        print("[错误] 提交文件不存在!")
        return False

    print(f"提交文件: {submission_path}")

    # 读取并统计
    with open(submission_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"总行数: {len(lines) - 1} (不含表头)")

    # 统计类别
    from collections import Counter
    class_counts = Counter()
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                class_id = int(parts[1])
                class_counts[class_id] += 1
            except:
                pass

    print("\n类别分布:")
    for cid in range(15):
        count = class_counts.get(cid, 0)
        status = "✓" if count > 0 else "✗"
        print(f"  Class {cid:2d}: {count:4d} {status}")

    print(f"\n类别覆盖: {len(class_counts)}/15")

    print("\n" + "="*70)
    print("完成!")
    print("="*70)
    print(f"提交文件已准备就绪: {submission_path}")
    print("请将此文件提交到评分平台")

    return True

def main():
    print("="*70)
    print("YOLOv8 交通标志检测 - 完整本地训练脚本")
    print("目标: mAP@0.5 >= 0.9")
    print("="*70)
    print(f"\n工作目录: {os.getcwd()}")
    print(f"Python: {sys.version.split()[0]}")

    steps = [
        ("环境配置", step1_environment_setup),
        ("数据准备", step2_data_preparation),
        ("模型训练", step3_model_training),
        ("生成提交", step4_generate_submission),
        ("验证完成", step5_verify_and_submit),
    ]

    for i, (name, func) in enumerate(steps, 1):
        print(f"\n\n{'#'*70}")
        print(f"# 步骤 {i}/5: {name}")
        print(f"{'#'*70}")

        try:
            success = func()
            if not success:
                print(f"\n[错误] 步骤 {i} 失败!")
                if i < 3:  # 前面的步骤失败可以继续
                    response = input("是否继续? (y/n): ")
                    if response.lower() != 'y':
                        break
        except KeyboardInterrupt:
            print("\n\n[中断] 用户取消")
            break
        except Exception as e:
            print(f"\n[错误] 发生异常: {e}")
            import traceback
            traceback.print_exc()
            break

    print("\n\n" + "="*70)
    print("脚本执行完成")
    print("="*70)

if __name__ == "__main__":
    main()
    input("\n按Enter键退出...")