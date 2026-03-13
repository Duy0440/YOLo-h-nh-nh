import cv2
import numpy as np

def load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def draw_boxes(frame, detections):
    """Draw bounding boxes and class labels on an image.

    Args:
        frame: BGR image (numpy array) that will be modified in-place.
        detections: list of dicts, each containing 'xyxy', 'name', and 'confidence'.

    Returns:
        The same frame object that was passed in (useful for chaining).
    """

    for d in detections:
        x1, y1, x2, y2 = d['xyxy']
        cls = d['name']
        conf = d['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
    return frame