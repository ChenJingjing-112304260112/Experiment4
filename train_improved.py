from ultralytics import YOLO
import sys

def train_model():
    """改进的训练脚本，使用更好的参数"""
    
    print("="*60)
    print("YOLOv8 Traffic Sign Detection - Improved Training")
    print("="*60)
    
    # 加载模型
    model = YOLO('yolov8s.pt')  # 使用更大的模型
    print("Loaded YOLOv8s model")
    
    # 改进的训练参数
    print("\nStarting training with improved parameters...")
    print("-"*40)
    print("Model: YOLOv8s (larger model for better accuracy)")
    print("Epochs: 50")
    print("Batch size: 8")
    print("Image size: 640x640")
    print("Optimizer: AdamW (better for small datasets)")
    print("Learning rate: 0.001")
    print("-"*40)
    
    try:
        results = model.train(
            data="第4次实验数据及提交格式/data.yaml",
            epochs=50,
            batch=8,
            imgsz=640,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            label_smoothing=0.0,
            nbs=64,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            save=True,
            save_period=-1,
            cache=False,
            image_weights=False,
            device=None,
            workers=8,
            project=None,
            name='traffic_signs_improved',
            exist_ok=False,
            pretrained=True,
            optimizer='AdamW',
            verbose=True,
            seed=0,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=True,  # 使用余弦学习率调度
            close_mosaic=10,
            resume=False,
            amp=True,
            fraction=1.0,
            profile=False,
            freeze=None,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
        )
        
        print("\nTraining completed successfully!")
        print(f"Best model saved to: runs/detect/traffic_signs_improved/weights/best.pt")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    train_model()