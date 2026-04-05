import cv2
import os
from src.detector import Detector
from src.face_recognizer import FaceRecognizer


def run_image(image_path):
    print("RUN IMAGE OK")

    detector = Detector()
    face_rec = FaceRecognizer()

    img = cv2.imread(image_path)

    if img is None:
        print("Cannot read image")
        return

    h, w = img.shape[:2]
    scale = w / 640
    thickness = max(2, int(3 * scale))
    font_scale = max(0.7, 0.8 * scale)

    # ===== YOLO =====
    detections = detector.predict(img)
    print(f"YOLO OK - found {len(detections)} people")
    total = len(detections)

    # ===== DRAW PERSON =====
    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(img, "Person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    # ===== 🔥 MULTI FACE =====
    faces = face_rec.recognize_faces(img)

    print(f"FACE OK - found {len(faces)} faces")

    # ===== DRAW FACE =====
    for f in faces:
        x1, y1, x2, y2 = f["box"]
        name = f["name"]
        score = f["score"]

        label = "Unknown" if name == "Unknown" else f"{name} {score:.1f}%"

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)

    # ===== TOTAL =====
    cv2.putText(img, f"Total: {total}", (20, int(40 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    # ===== SAVE =====
    os.makedirs("output", exist_ok=True)
    save_path = "output/result.jpg"
    cv2.imwrite(save_path, img)

    print(f"Saved: {save_path}")

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()