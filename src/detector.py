import cv2
import numpy as np
from ultralytics import YOLO


class Detector:
    """Simple wrapper for running YOLO object detection.

    This class keeps the interface small so it is easy to explain:
      - `predict(frame)` returns a list of detections
      - `save(frame, out_path)` writes a copy of the frame with boxes/labels

    The code does not attempt to optimize performance (e.g., batching) so it is
    easy to read and reason about.
    """

    def __init__(self, model_path: str = 'yolov8n.pt', conf: float = 0.25):
        # Load pre-trained YOLO weights. This can take a second.
        self.model = YOLO(model_path)
        self.confidence = conf

    def predict(self, frame: np.ndarray):
        """Run inference on one frame and return detections.

        Returns a list of dicts, each with:
          - "xyxy": (xmin, ymin, xmax, ymax)
          - "confidence": score from 0..1
          - "class": integer class id
          - "name": human-readable class name (e.g., "person")
        """

        results = self.model.predict(source=frame, conf=self.confidence, verbose=False)
        if len(results) == 0:
            return []

        # ultralytics returns a list of Result objects; we take the first (only) one.
        res = results[0]

        dets = []
        for *box, conf, cls in res.boxes.data.tolist():
            xmin, ymin, xmax, ymax = box
            dets.append({
                "xyxy": (int(xmin), int(ymin), int(xmax), int(ymax)),
                "confidence": float(conf),
                "class": int(cls),
                "name": self.model.names[int(cls)],
            })
        return dets

    def save(self, frame: np.ndarray, out_path: str, dets=None):
        """Draw boxes+labels on a frame and write the result to disk."""

        if dets is None:
            dets = self.predict(frame)

        # We draw onto the original frame in-place.
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            cls = d["name"]
            conf = d["confidence"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{cls} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        cv2.imwrite(out_path, frame)
