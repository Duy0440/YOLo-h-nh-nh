import cv2
import os
from src.detector import Detector
from src.face_recognizer import FaceRecognizer
import time

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
    print(f"YOLO OK - found {len(detections)} people")

    total = len(detections)

    final_results = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        if x2 <= x1 or y2 <= y1:
            continue

        person_crop = img[y1:y2, x1:x2]

        
        h_crop = person_crop.shape[0]
        person_crop = person_crop[0:int(h_crop * 0.5), :]

        faces = face_rec.recognize_faces(person_crop)

        
        best_face = None
        min_dist = 999999

        px_center = (x2 - x1) // 2
        py_center = int((y2 - y1) * 0.25)

        for f in faces:
            fx1, fy1, fx2, fy2 = f["box"]

            fx_center = (fx1 + fx2) // 2
            fy_center = (fy1 + fy2) // 2

            dist = (fx_center - px_center) ** 2 + (fy_center - py_center) ** 2

            if dist < min_dist:
                min_dist = dist
                best_face = f

       
        if best_face is not None and best_face["name"] != "Unknown" and best_face["score"] >= 65:
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

   
    for res in final_results:
        x1, y1, x2, y2 = res["box"]

        # vẽ mặt
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(img, "Person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

       
        if res["name"] is not None:
            name = res["name"]
            score = res["score"]

            label = f"{name} {score:.1f}%"

            fx1, fy1, fx2, fy2 = res["face_box"]

            
            fx1 += x1
            fy1 += y1
            fx2 += x1
            fy2 += y1

            # box mặt
            cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (255, 0, 0), thickness)

            
            cv2.putText(img, label, (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)

    
    cv2.putText(img, f"Total: {total}", (20, int(40 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    os.makedirs("output", exist_ok=True)

    timestamp = int(time.time())
    save_path = f"output/result_{timestamp}.jpg"

    cv2.imwrite(save_path, img)
    print(f"Saved: {save_path}")

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()