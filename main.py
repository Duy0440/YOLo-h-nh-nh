import argparse
import os
import cv2

from src.detector import Detector
from src.face_recognizer import FaceRecognizer


def process_image(input_path, output_path, model_path, conf):

    detector = Detector(model_path, conf)
    face_rec = FaceRecognizer()

    img = cv2.imread(input_path)

    if img is None:
        print("Cannot read image")
        return

    detections = detector.predict(img)

    person_count = 0

    for d in detections:

        if d["name"] != "person":
            continue

        x1, y1, x2, y2 = d["box"]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        person_crop = img[y1:y2, x1:x2]

        if person_crop.size == 0:
            continue

        name, conf_face = face_rec.recognize(person_crop)
        print("Detected face:", name, conf_face)

        label = f"{name} {conf_face:.2f}"

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

        person_count += 1

    cv2.putText(
        img,
        f"Total people: {person_count}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    cv2.imwrite(output_path, img)

    print("Total people detected:", person_count)
    print("Saved result:", output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)

    args = parser.parse_args()

    process_image(
        args.input,
        args.output,
        args.model,
        args.conf
    )