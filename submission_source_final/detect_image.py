import cv2
import os
import time
from src.detector import Detector
from src.face_recognizer import FaceRecognizer


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def filter_person_detections(detections, img_w, img_h):
    filtered = []

    for det in sorted(detections, key=lambda item: item[4], reverse=True):
        x1, y1, x2, y2, conf, cls = det
        box_w = x2 - x1
        box_h = y2 - y1
        area = box_w * box_h

        
        if box_w < 55 or box_h < 110:
            continue
        if area < img_w * img_h * 0.03:
            continue

        is_duplicate = False
        for kept in filtered:
            if box_iou((x1, y1, x2, y2), kept[:4]) > 0.5:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(det)

    return filtered


def pick_best_face(faces, person_width, person_height):
    if not faces:
        return None

    px_center = person_width // 2
    py_center = int(person_height * 0.25)

    best_face = None
    best_score = None

    for face in faces:
        fx1, fy1, fx2, fy2 = face["box"]
        fw = fx2 - fx1
        fh = fy2 - fy1

        if fw <= 0 or fh <= 0:
            continue

        fx_center = (fx1 + fx2) // 2
        fy_center = (fy1 + fy2) // 2

       
        if abs(fx_center - px_center) > person_width * 0.25:
            continue
        if fy_center > person_height * 0.7:
            continue

        center_dist = (fx_center - px_center) ** 2 + (fy_center - py_center) ** 2
        distance = face.get("distance", 1.0)
        candidate_score = center_dist + distance * 5000

        if best_score is None or candidate_score < best_score:
            best_score = candidate_score
            best_face = face

    return best_face


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

    detections = detector.predict(img)
    detections = filter_person_detections(detections, w, h)
    print(f"YOLO OK - kept {len(detections)} people after filtering")

    final_results = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        if x2 <= x1 or y2 <= y1:
            continue

        person_crop = img[y1:y2, x1:x2]
        person_h, person_w = person_crop.shape[:2]

        person_crop = person_crop[0:int(person_h * 0.5), :]
        crop_h, crop_w = person_crop.shape[:2]

        faces = face_rec.recognize_faces(person_crop)
        best_face = pick_best_face(faces, crop_w, crop_h)

        if (
            best_face is not None
            and best_face["name"] != "Unknown"
            and best_face["score"] >= 70
            and best_face.get("distance", 1.0) <= 0.28
        ):
            final_results.append({
                "box": (x1, y1, x2, y2),
                "name": best_face["name"],
                "score": best_face["score"],
                "face_box": best_face["box"]
            })
        else:
            final_results.append({
                "box": (x1, y1, x2, y2),
                "name": None
            })

    total = len(final_results)

    for res in final_results:
        x1, y1, x2, y2 = res["box"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(
            img,
            "Person",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
        )

        if res["name"] is not None:
            name = res["name"]
            score = res["score"]
            label = f"{name} {score:.1f}%"

            fx1, fy1, fx2, fy2 = res["face_box"]
            fx1 += x1
            fy1 += y1
            fx2 += x1
            fy2 += y1

            cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (255, 0, 0), thickness)
            cv2.putText(
                img,
                label,
                (fx1, fy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 0, 0),
                thickness,
            )

    cv2.putText(
        img,
        f"Total: {total}",
        (20, int(40 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 255),
        thickness,
    )

    os.makedirs("output", exist_ok=True)

    timestamp = int(time.time())
    save_path = f"output/result_{timestamp}.jpg"

    cv2.imwrite(save_path, img)
    print(f"Saved: {save_path}")

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
