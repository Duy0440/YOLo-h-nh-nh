"""Helper: convert pixel bbox to YOLO format line.

Usage:
  python scripts/make_yolo_label.py --image input/images/your.jpg --xmin 100 --ymin 80 --xmax 400 --ymax 450 --class 0

This prints the line you can paste into a .txt label file.
"""

import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description="Convert pixel bbox to YOLO label line")
    parser.add_argument("--image", required=True, help="Path to the image")
    parser.add_argument("--xmin", type=int, required=True)
    parser.add_argument("--ymin", type=int, required=True)
    parser.add_argument("--xmax", type=int, required=True)
    parser.add_argument("--ymax", type=int, required=True)
    parser.add_argument("--class", type=int, required=True, dest="cls", help="Class ID (0=male, 1=female)")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    h, w = img.shape[:2]
    x_center = (args.xmin + args.xmax) / 2.0 / w
    y_center = (args.ymin + args.ymax) / 2.0 / h
    width = (args.xmax - args.xmin) / w
    height = (args.ymax - args.ymin) / h

    print(f"{args.cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")


if __name__ == "__main__":
    main()
