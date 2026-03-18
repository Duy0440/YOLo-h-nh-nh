from ultralytics import YOLO


class Detector:

    def __init__(self, model_path, conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, image):

        results = []

        # chạy YOLO
        output = self.model(image, conf=self.conf, verbose=False)

        # lấy kết quả đầu tiên
        boxes = output[0].boxes

        if boxes is None:
            return results

        for box in boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # chỉ lấy người (class 0 = person)
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            results.append({
                "name": "person",
                "box": (x1, y1, x2, y2),
                "conf": conf
            })

        return results