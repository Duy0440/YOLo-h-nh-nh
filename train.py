"""Minimal training script for a YOLOv8 object detection dataset."""

from ultralytics import YOLO


def train_model(data_yaml: str,
                epochs: int = 50,
                batch: int = 16,
                model: str = 'yolov8n.pt',
                imgsz: int = 640,
                project: str = 'models',
                name: str = 'exp'):
    yolo = YOLO(model)
    yolo.train(data=data_yaml,
               epochs=epochs,
               batch=batch,
               imgsz=imgsz,
               project=project,
               name=name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train YOLOv8 detector')
    parser.add_argument('--data', type=str, required=True,
                        help='path to dataset .yaml file')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='base model to fine-tune')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--project', type=str, default='models')
    parser.add_argument('--name', type=str, default='exp')
    args = parser.parse_args()
    train_model(args.data, args.epochs, args.batch, args.model,
                args.imgsz, args.project, args.name)
