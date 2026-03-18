import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import sys

class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass

sys.stderr = DevNull()

import argparse
import cv2

from src.detector import Detector
from src.face_recognizer import FaceRecognizer
from src.counter import Counter

# ================= IMAGE =================
def process_image(input_path, output_path, model_path, conf):

    detector = Detector(model_path, conf)
    face_rec = FaceRecognizer()

    img = cv2.imread(input_path)

    if img is None:
        print("Cannot read image")
        return

    # 👉 phóng to nhẹ để nhận mặt tốt hơn
    img = cv2.resize(img, None, fx=1.3, fy=1.3)

    # ===== YOLO đếm người =====
    detections = detector.predict(img)

    total = sum(1 for obj in detections if obj["name"] == "person")

    # ===== scale =====
    h, w = img.shape[:2]
    scale = w / 640
    thickness = max(1, int(2 * scale))
    font_scale = max(0.5, 0.6 * scale)

    # ===== FACE =====
    faces = face_rec.detect_and_recognize(img)

    for f in faces:
        x1, y1, x2, y2 = f["box"]
        name = f["name"]
        score = f["score"]

        label = "Unknown" if name == "Unknown" else f"{name} {round(score,1)}%"

        # 👉 LUÔN vẽ khung (không skip)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), thickness)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0,255,0), thickness)

    # ===== TOTAL =====
    cv2.putText(img, f"Total: {total}", (10, int(40 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0,0,255), thickness)

    cv2.imwrite(output_path, img)
    print("Done image")


# ================= VIDEO =================
def process_video(input_path, output_path, model_path, conf):

    detector = Detector(model_path, conf)
    face_rec = FaceRecognizer()
    counter = Counter()   # 🔥 THÊM

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Cannot open video")
        return

    out = None
    frame_count = 0
    faces = []

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ===== YOLO =====
        detections = detector.predict(frame)

        # 🔥 TRACK + COUNT (QUAN TRỌNG)
        objects = counter.update(detections)
        total = counter.total()

        # ===== scale =====
        h, w = frame.shape[:2]
        scale = w / 640
        thickness = max(1, int(2 * scale))
        font_scale = max(0.5, 0.6 * scale)

        # 🔥 FACE mỗi 10 frame
        if frame_count % 10 == 0:
            faces = face_rec.detect_and_recognize(frame)

        # ===== DRAW FACE =====
        for f in faces:
            x1, y1, x2, y2 = f["box"]
            name = f["name"]
            score = f["score"]

            label = "Unknown" if name == "Unknown" else f"{name} {round(score,1)}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), thickness)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0,255,0), thickness)

        # ===== DRAW PERSON BOX + ID =====
        for oid, centroid in objects.items():
            cx, cy = centroid
            cv2.putText(frame, f"ID {oid}", (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255,0,0), thickness)

        # ===== TOTAL (CHUẨN DEMO) =====
        cv2.putText(frame, f"Total: {total}", (10, int(40 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0,0,255), thickness)

        # ===== WRITE =====
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20, (w, h))

        out.write(frame)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    print("Done video")


# ================= MAIN =================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--mode", choices=["image", "video"], default="image")

    args = parser.parse_args()

    if args.mode == "image":
        process_image(args.input, args.output, args.model, args.conf)
    else:
        process_video(args.input, args.output, args.model, args.conf)