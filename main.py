import argparse
import os
from typing import Optional

import cv2

from src.detector import Detector
from src.counter import Counter
from src.utils import draw_boxes


def process_video(
    input_path: str,
    output_dir: str,
    model_path: str,
    conf: float = 0.25,
    target_class: Optional[str] = None,
    draw_bboxes: bool = True,
):
    """Run detection frame-by-frame on a video and write an annotated output video."""

    # Ensure output folder exists (like a "runs/" folder in other YOLO repos)
    os.makedirs(output_dir, exist_ok=True)

    # Detector is responsible for running YOLO inference on each frame
    det = Detector(model_path, conf=conf)

    # Counter tracks unique detected objects across frames (same as before)
    cnt = Counter()

    cap = cv2.VideoCapture(input_path)

    # Prepare video writer (same resolution + fps as input)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(output_dir, os.path.basename(input_path))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Detect objects in current frame
        detections = det.predict(frame)
        if target_class:
            # Only keep detections of the desired class (e.g., "person").
            detections = [d for d in detections if d["name"] == target_class]

        # 2) Update tracker counter based on detections
        cnt.update(detections)

        # 3) Optionally draw bounding boxes + labels on frame
        vis = frame
        if draw_bboxes:
            vis = draw_boxes(frame, detections)

        # (No overlay text; counts are printed to terminal instead)
        pass

        # Print to terminal on every 30th frame to avoid flooding
        if frame_id % 30 == 0:
            class_counts = {}
            for d in detections:
                class_counts[d["name"]] = class_counts.get(d["name"], 0) + 1
            print(
                f"Frame {frame_id}: total={len(detections)}, "
                + ", ".join(f"{k}={v}" for k, v in class_counts.items())
            )

        writer.write(vis)
        frame_id += 1

    cap.release()
    writer.release()

    print(f"Finished processing video. Output saved to: {out_path}")
    print(f"Total unique objects seen (tracked IDs): {cnt.totals()}")


def process_image(
    input_path: str,
    output_path: str,
    model_path: str,
    conf: float = 0.25,
    target_class: Optional[str] = None,
    draw_bboxes: bool = True,
):
    """Run detection on a single image and save the annotated result."""

    det = Detector(model_path, conf=conf)
    img = cv2.imread(input_path)
    if img is None:
        raise IOError(f"Could not read image: {input_path}")

    detections = det.predict(img)
    if target_class:
        detections = [d for d in detections if d["name"] == target_class]

    # Build a class -> count summary for terminal output
    class_counts = {}
    for d in detections:
        class_counts[d["name"]] = class_counts.get(d["name"], 0) + 1

    print(
        "Counts:",
        "total=", len(detections),
        ", ".join(f"{k}={v}" for k, v in class_counts.items()),
    )

    vis = img
    if draw_bboxes:
        vis = draw_boxes(img, detections)

    cv2.imwrite(output_path, vis)
    print(f"Saved annotated image to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect and count objects')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='path to image or video',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='output directory (for video) or output file (for image)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/yolov8n.pt',
        help='path to YOLO model weights (for example: models/yolov8n.pt)',
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='confidence threshold (higher -> fewer boxes)',
    )
    parser.add_argument(
        '--target-class',
        type=str,
        default=None,
        help='only draw/count this class name (e.g., person)',
    )
    parser.add_argument(
        '--no-draw',
        action='store_true',
        dest='no_draw',
        help='do not draw bounding boxes on output (useful to avoid clutter)',
    )

    args = parser.parse_args()

    # If input is a video file or folder, treat it as video processing
    if os.path.isdir(args.input) or args.input.lower().endswith(('.mp4', '.avi', '.mkv')):
        process_video(
            args.input,
            args.output,
            args.model,
            conf=args.conf,
            target_class=args.target_class,
            draw_bboxes=not args.no_draw,
        )
    else:
        process_image(
            args.input,
            args.output,
            args.model,
            conf=args.conf,
            target_class=args.target_class,
            draw_bboxes=not args.no_draw,
        )
