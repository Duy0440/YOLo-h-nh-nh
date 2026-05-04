from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="models/yolov8n.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, frame):

        h, w = frame.shape[:2]

        results = self.model(frame, conf=self.conf)[0]

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:

            cls = int(box.cls[0])

            
            if cls != 0:
                continue

            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

           
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            detections.append([x1, y1, x2, y2, conf, cls])

        return detections