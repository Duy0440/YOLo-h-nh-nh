"""Simple script to train a gender-detection YOLO model.

This script assumes you have prepared a dataset in YOLO format:
 - Images in `train/` and `val/`
 - Each image has a corresponding `.txt` label file with bounding boxes
   and class IDs (0=male, 1=female)

See `data_gender.yaml` for the config format.

Usage example:
    python train_gender.py --data data_gender.yaml --epochs 20 --model yolov8n.pt

After training, the best weights will be saved in `runs/train/<name>/weights/best.pt`.
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model for gender detection")
    parser.add_argument(
        "--data",
        type=str,
        default="data_gender.yaml",
        help="path to dataset yaml (train/val paths + nc + names)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="base YOLO weights to start from (e.g. yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="batch size",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="gender",
        help="name for the training run (output folder)",
    )
    args = parser.parse_args()

    model = YOLO(args.model)

    # Run training
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        project="runs/train",
    )

    print("Training complete. Weights are saved under runs/train/{}/weights".format(args.name))


if __name__ == "__main__":
    main()
