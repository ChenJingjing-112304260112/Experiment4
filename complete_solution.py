import sys
import csv
import os
from pathlib import Path

# Add debugging output to file
log_file = open('training_log.txt', 'w')
sys.stdout = log_file
sys.stderr = log_file

print("="*60)
print("COMPLETE TRAINING AND INFERENCE SCRIPT")
print("="*60)
print("\nStep 1: Importing YOLO...")

try:
    from ultralytics import YOLO
    print("✓ YOLO imported successfully")
    
    print("\nStep 2: Loading model...")
    model = YOLO("yolov8n.pt")
    print("✓ Model loaded")
    
    print("\nStep 3: Starting training...")
    print("Data path:", os.path.abspath("第4次实验数据及提交格式/data.yaml"))
    
    # Train the model
    results = model.train(
        data="第4次实验数据及提交格式/data.yaml",
        epochs=5,
        batch=1,
        imgsz=416,
        verbose=True,
        device='cpu',
        workers=0,
        name='traffic_signs_complete'
    )
    
    print("\n✓ Training completed")
    print("Results:", results)
    
    # Check for weights
    weights_dir = "runs/detect/traffic_signs_complete/weights"
    print("\nChecking weights directory:", weights_dir)
    
    if os.path.exists(weights_dir):
        files = os.listdir(weights_dir)
        print("Files found:", files)
        
        if "best.pt" in files:
            model_path = os.path.join(weights_dir, "best.pt")
            print("Using best.pt")
        elif "last.pt" in files:
            model_path = os.path.join(weights_dir, "last.pt")
            print("Using last.pt")
        else:
            model_path = "yolov8n.pt"
            print("Using original yolov8n.pt")
    else:
        model_path = "yolov8n.pt"
        print("Weights directory not found, using original model")
    
    print("\nStep 4: Running inference...")
    trained_model = YOLO(model_path)
    
    test_dir = Path("第4次实验数据及提交格式/test/images")
    image_paths = sorted([p for p in test_dir.iterdir() if p.is_file()])
    print(f"Found {len(image_paths)} test images")
    
    # Generate submission
    output_path = Path("submission.csv")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()
        
        count = 0
        for result in trained_model.predict(source=[str(p) for p in image_paths], conf=0.01, save=False, verbose=False):
            image_id = Path(result.path).name
            if result.boxes is not None:
                for box in result.boxes:
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    writer.writerow({
                        "image_id": image_id,
                        "class_id": int(box.cls[0].item()),
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "confidence": float(box.conf[0].item()),
                    })
                    count += 1
        
        print(f"Total detections: {count}")
    
    print(f"\n✓ Submission file saved to: {output_path}")
    print("\nDone!")
    
except Exception as e:
    print("\n✗ ERROR:", str(e))
    import traceback
    traceback.print_exc()

finally:
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("Training log saved to training_log.txt")