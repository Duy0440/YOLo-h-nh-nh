from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="models/yolov8n.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, frame):
        results = self.model(frame, conf=self.conf)[0]

        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])

            if cls != 0:  # person
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "name": "person",
                "box": (x1, y1, x2, y2)
            })

        return detections