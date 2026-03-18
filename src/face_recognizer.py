import os
import cv2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from deepface import DeepFace
from mtcnn import MTCNN


class FaceRecognizer:

    def __init__(self, db_path="faces"):
        self.db_path = db_path
        self.detector = MTCNN()

    def detect_and_recognize(self, image):

        results = []

        try:
            faces = self.detector.detect_faces(image)

            for face in faces:

                x, y, w, h = face["box"]

                x = max(0, x)
                y = max(0, y)
                x2 = x + w
                y2 = y + h

                face_crop = image[y:y2, x:x2]

                if face_crop.size == 0:
                    continue

                face_crop = cv2.resize(face_crop, (224, 224))

                name = "Unknown"
                score = 0

                dfs = DeepFace.find(
                    img_path=face_crop,
                    db_path=self.db_path,
                    enforce_detection=False,
                    silent=True
                )

                if len(dfs) > 0 and len(dfs[0]) > 0:

                    best = dfs[0].iloc[0]

                    path = best["identity"]
                    candidate_name = os.path.basename(os.path.dirname(path))

                    distance = best["distance"]

                    # 🔥 NGƯỠNG CHUẨN (đỡ nhận sai)
                    if distance < 0.55:
                        score = (1 - distance) * 100

                        if score >= 60:
                            name = candidate_name
                        else:
                            name = "Unknown"
                            score = 0

                results.append({
                    "box": (x, y, x2, y2),
                    "name": name,
                    "score": score
                })

        except Exception as e:
            print("Face error:", e)

        return results