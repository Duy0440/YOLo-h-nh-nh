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

            if cls != 0:  # chỉ lấy person
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            # format chuẩn cho tracker + code mày
            detections.append([x1, y1, x2, y2, conf, cls])

        return detections