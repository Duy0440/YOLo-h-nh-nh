from ultralytics import YOLO

class Tracker:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml"
        )[0]

        objects = []

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])

                if cls != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = -1

                objects.append((track_id, x1, y1, x2, y2))

        return objects