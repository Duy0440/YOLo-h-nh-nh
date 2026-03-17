from ultralytics import YOLO


class Detector:

    def __init__(self, model_path="models/yolov8n.pt", conf=0.25):

        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, frame):

        results = self.model(frame, conf=self.conf)

        detections = []

        for r in results:

            boxes = r.boxes

            for box in boxes:

                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]

                x1, y1, x2, y2 = box.xyxy[0]

                detections.append({
                    "name": name,
                    "box": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    )
                })

        return detections