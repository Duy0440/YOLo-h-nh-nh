from deepface import DeepFace
import os

class FaceRecognizer:
    def __init__(self, db_path="faces"):
        self.db_path = db_path

    def detect_and_recognize(self, frame):
        results = []

        try:
            dfs = DeepFace.find(
                img_path=frame,
                db_path=self.db_path,
                enforce_detection=False,
                detector_backend="retinaface",   # 🔥 detect chuẩn hơn
                model_name="Facenet512",
                silent=True   # 🔥 tắt log rác
            )

            if len(dfs) > 0 and len(dfs[0]) > 0:
                for _, row in dfs[0].iterrows():

                    identity = row["identity"]

                    # 👉 lấy tên thư mục (cross-platform, không lỗi \ hay /)
                    name = os.path.basename(os.path.dirname(identity))

                    x = int(row["source_x"])
                    y = int(row["source_y"])
                    w = int(row["source_w"])
                    h = int(row["source_h"])

                    score = (1 - row["distance"]) * 100

                    # 👉 lọc độ tin cậy thấp
                    if score < 60:
                        name = "Unknown"

                    results.append({
                        "name": name,
                        "box": (x, y, x + w, y + h),
                        "score": score
                    })

        except Exception as e:
            print("Face error:", e)

        return results