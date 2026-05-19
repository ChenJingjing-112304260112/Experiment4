"""
快速评估模型性能
"""

from ultralytics import YOLO
import sys

def quick_eval():
    model = YOLO("runs/detect/traffic_signs_complete/weights/best.pt")
    
    # 评估验证集
    print("Evaluating on validation set...", file=sys.stderr)
    metrics = model.val(
        data="第4次实验数据及提交格式/data.yaml",
        split="val",
        verbose=True,
        device="cpu"
    )
    
    # 打印结果
    print("\nValidation Metrics:", file=sys.stderr)
    print(f"  mAP50: {metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}", file=sys.stderr)
    print(f"  mAP50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}", file=sys.stderr)
    print(f"  Precision: {metrics.results_dict.get('metrics/precision(B)', 'N/A')}", file=sys.stderr)
    print(f"  Recall: {metrics.results_dict.get('metrics/recall(B)', 'N/A')}", file=sys.stderr)

if __name__ == "__main__":
    quick_eval()