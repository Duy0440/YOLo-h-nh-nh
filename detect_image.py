import cv2
import os
from src.detector import Detector
from src.face_recognizer import FaceRecognizer


def run_image(image_path):
    print("RUN IMAGE OK")

    try:
        detector = Detector()
        face_rec = FaceRecognizer()

        img = cv2.imread(image_path)

        if img is None:
            print("Cannot read image")
            return

        # ===== SCALE cho đẹp =====
        h, w = img.shape[:2]
        scale = w / 640
        thickness = max(2, int(3 * scale))
        font_scale = max(0.7, 0.8 * scale)

        # ===== YOLO detect =====
        detections = detector.predict(img)
        print(f"YOLO OK - found {len(detections)} people")

        # ===== VẼ KHUNG NGƯỜI =====
        for det in detections:
            x1, y1, x2, y2, conf, cls = det

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.putText(img, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        total = len(detections)

        # ===== FACE (CROP THEO NGƯỜI) =====
        faces = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det

            if x2 <= x1 or y2 <= y1:
                continue

            # clamp tránh lỗi
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            person_crop = img[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            results = face_rec.detect_and_recognize(person_crop)

            for r in results:
                fx1, fy1, fx2, fy2 = r["box"]

                faces.append({
                    "name": r["name"],
                    "score": r["score"],
                    "box": (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)
                })

        print(f"FACE OK - found {len(faces)} faces")

        # ===== VẼ FACE =====
        for f in faces:
            x1, y1, x2, y2 = f["box"]
            name = f["name"]
            score = f["score"]

            label = "Unknown" if name == "Unknown" else f"{name} {round(score,1)}%"

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)

        # ===== TOTAL =====
        cv2.putText(img, f"Total: {total}", (20, int(40 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

        # ===== SAVE =====
        os.makedirs("output", exist_ok=True)

        filename = os.path.basename(image_path)
        name, _ = os.path.splitext(filename)

        save_path = f"output/{name}_result.jpg"

        cv2.imwrite(save_path, img)

        print(f"Saved: {save_path}")

        # ===== SHOW =====
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("ERROR:", e)