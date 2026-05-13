from ultralytics import YOLO
import argparse
import sys

def main():
    print("Starting training script...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model")
    parser.add_argument("--name", default="traffic_signs", help="Run name")
    args = parser.parse_args()

    print(f"Arguments: {args}", flush=True)
    try:
        print("Loading model...", flush=True)
        model = YOLO(args.model)
        print("Model loaded successfully", flush=True)
        print("Starting training...", flush=True)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            name=args.name,
            augment=True,
            patience=5,
            verbose=True,
            device='cpu',
            workers=0
        )
        print("Training completed!", flush=True)
        print(f"Best model saved to: runs/detect/{args.name}/weights/best.pt", flush=True)
    except Exception as e:
        print(f"Error during training: {str(e)}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    main()