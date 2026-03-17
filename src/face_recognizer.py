import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from deepface import DeepFace


class FaceRecognizer:

    def __init__(self, db_path="faces"):
        self.db_path = db_path

    def recognize(self, image):

        try:

            dfs = DeepFace.find(
                img_path=image,
                db_path=self.db_path,
                enforce_detection=False,
                silent=True
            )

            if len(dfs) > 0 and len(dfs[0]) > 0:

                best = dfs[0].iloc[0]

                path = best["identity"]

                name = os.path.basename(os.path.dirname(path))

                distance = best["distance"]

                confidence = 1 - distance

                return name, confidence

        except Exception as e:
            print("Face error:", e)

        return "Unknown", 0