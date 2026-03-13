"""Create directory structure and sample label files for a gender detection dataset.

This is a helper script: it creates the expected folder layout, and can optionally
copy a sample image into the train/val folders and create a placeholder label file.

Usage:
    python scripts/make_gender_dataset_structure.py --sample input/images/hinhanh1.png

It does NOT generate real bounding boxes; it only creates directories and an example
label file that you should replace with real labels.
"""

import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description="Create gender dataset folder structure")
    parser.add_argument(
        "--base",
        type=str,
        default="input/images/gender",
        help="Base folder for gender dataset",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Optional sample image to copy into train/val for a quick start",
    )
    args = parser.parse_args()

    train_dir = os.path.join(args.base, "train")
    val_dir = os.path.join(args.base, "val")

    # YOLO expects images and labels to be in parallel folders.
    # E.g., images in input/images/gender/train and labels in input/labels/gender/train
    labels_base = os.path.join("input", "labels", os.path.basename(os.path.normpath(args.base)))
    train_label_dir = os.path.join(labels_base, "train")
    val_label_dir = os.path.join(labels_base, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    if args.sample:
        if not os.path.isfile(args.sample):
            raise FileNotFoundError(f"Sample image not found: {args.sample}")

        # Copy sample into both train and val so you have something to start with.
        sample_name = os.path.basename(args.sample)
        shutil.copyfile(args.sample, os.path.join(train_dir, sample_name))
        shutil.copyfile(args.sample, os.path.join(val_dir, sample_name))

        # Create placeholder label files (same name, .txt) that the user should update.
        # Here we use a dummy bounding box (center, 0.2x0.2) for class 0.
        label_text = "0 0.5 0.5 0.2 0.2\n"
        for d in (train_label_dir, val_label_dir):
            label_path = os.path.join(d, os.path.splitext(sample_name)[0] + ".txt")
            with open(label_path, "w") as f:
                f.write(label_text)

    print("Gender dataset structure created:")
    print(f"  train: {train_dir}")
    print(f"  val:   {val_dir}")
    if args.sample:
        print("Sample image and placeholder label files created. Please update the .txt labels with correct bounding boxes.")


if __name__ == "__main__":
    main()
