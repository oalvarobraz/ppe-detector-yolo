import argparse
import sys
from ultralytics import YOLO
import traceback


def train(model, data, epochs, patience, batch, device, name, imgsz, project):
    model.train(data=data, epochs=epochs, patience=patience, 
                batch=batch, device=device, name=name, imgsz=imgsz, project=project)
    best_path = model.trainer.best
    return best_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/construction-safety/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="ppe_v2")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Pasta onde salvar os resultados (no Colab, use uma pasta do Google Drive)")

    args = parser.parse_args()

    try:
        model = YOLO(args.model)
        best_path = train(model=model, data=args.data, epochs=args.epochs, patience=args.patience,
              batch=args.batch, device=args.device, name=args.name, imgsz=args.imgsz, project=args.project)

    except Exception as e:
        print(f"Error during training model: {e}")
        traceback.print_exc()
        sys.exit(1)

    metrics = YOLO(best_path).val(data=args.data, split="test", imgsz=args.imgsz, batch=1, rect=False,
                                  project=args.project, name=f"{args.name}_test")

    print(f"mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")
    for i, c in enumerate(metrics.box.ap_class_index):
        print(f"  {metrics.names[c]}: {metrics.box.ap50[i]:.3f}")


    print(f"best_path: {best_path}")