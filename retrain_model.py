"""
完整的重新训练脚本
使用YOLOv8s模型和更好的训练参数
"""

from ultralytics import YOLO
import argparse
import sys

def train_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", default="yolov8s.pt", help="Base model")
    args = parser.parse_args()
    
    print("="*60)
    print("YOLOv8 Traffic Sign Detection - Retraining")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print("="*60)
    
    try:
        # 加载模型
        model = YOLO(args.model)
        print("Model loaded successfully")
        
        # 训练
        print("\nStarting training...")
        results = model.train(
            data="第4次实验数据及提交格式/data.yaml",
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            optimizer='AdamW',
            lr0=0.001,
            cos_lr=True,
            name='traffic_signs_retrained',
            verbose=True,
            device='cpu'
        )
        
        print("\nTraining completed!")
        print(f"Best model saved to: runs/detect/traffic_signs_retrained/weights/best.pt")
        
        # 验证模型
        print("\nValidating model...")
        metrics = model.val()
        print(f"\nValidation Results:")
        print(f"  mAP50: {metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    train_model()