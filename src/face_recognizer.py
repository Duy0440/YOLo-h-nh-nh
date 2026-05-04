from deepface import DeepFace
import os
import numpy as np
import cv2


class FaceRecognizer:

    def __init__(self, db_path="faces"):
        self.db_path = db_path
        self.database = []

        print("Loading face database...")

        for person in os.listdir(db_path):
            person_path = os.path.join(db_path, person)

            if not os.path.isdir(person_path):
                continue

            for img_name in os.listdir(person_path):

                img_path = os.path.join(person_path, img_name)

                try:
                    
                    faces = DeepFace.extract_faces(
                        img_path=img_path,
                        detector_backend="retinaface",
                        enforce_detection=False,
                        align=True
                    )

                    if len(faces) == 0:
                        continue

                    face_img = faces[0]["face"]
                    face_img = (face_img * 255).astype("uint8")
                    face_img = cv2.resize(face_img, (160, 160))

                    embedding = DeepFace.represent(
                        img_path=face_img,
                        model_name="Facenet512",
                        enforce_detection=False
                    )[0]["embedding"]

                    embedding = np.array(embedding)
                    embedding = embedding / np.linalg.norm(embedding)

                    self.database.append({
                        "name": person,
                        "embedding": embedding
                    })

                except:
                    pass

        print(f"Loaded {len(self.database)} face embeddings")

    def cosine_distance(self, a, b):
        return 1 - np.dot(a, b)

    def recognize_faces(self, img):

        results = []

        try:

            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend="retinaface",
                enforce_detection=False,
                align=True
            )

            for face in faces:

                
                face_img = face["face"]
                face_img = (face_img * 255).astype("uint8")
                face_img = cv2.resize(face_img, (160, 160))

                area = face["facial_area"]
                x, y, w, h = area["x"], area["y"], area["w"], area["h"]

                embedding = DeepFace.represent(
                    img_path=face_img,
                    model_name="Facenet512",
                    enforce_detection=False
                )[0]["embedding"]

                embedding = np.array(embedding)
                embedding = embedding / np.linalg.norm(embedding)

                best_name = "Unknown"
                best_distance = 999

                for db in self.database:

                    dist = self.cosine_distance(embedding, db["embedding"])

                    if dist < best_distance:
                        best_distance = dist
                        best_name = db["name"]

                score = (1 - best_distance) * 100

                # threshold 
                if best_distance > 0.45:
                    best_name = "Unknown"

                results.append({
                    "name": best_name,
                    "box": (x, y, x + w, y + h),
                    "score": round(score, 2),
                    "distance": float(best_distance)
                })

               
                print(f"[DEBUG] {best_name} | distance: {best_distance:.3f}")

        except Exception as e:
            print("Face error:", e)

        return results
