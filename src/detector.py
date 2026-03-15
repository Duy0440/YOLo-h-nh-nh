import cv2
import numpy as np
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolov8n.pt', conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, frame: np.ndarray):
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        if not results:
            return []
        dets = []
        for *box, conf, cls in results[0].boxes.data.tolist():
            xmin, ymin, xmax, ymax = map(int, box)
            dets.append({
                'xyxy': (xmin, ymin, xmax, ymax),
                'confidence': float(conf),
                'class': int(cls),
                'name': self.model.names[int(cls)]
            })
        return dets

    def save(self, frame: np.ndarray, out_path: str, dets=None):
        if dets is None:
            dets = self.predict(frame)
        for d in dets:
            x1, y1, x2, y2 = d['xyxy']
            cls, conf = d['name'], d['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
        cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1-10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imwrite(out_path, frame)