import cv2
import os
from tracker import Tracker
from face_recognizer import FaceRecognizer

def run_video(video_path):
    tracker = Tracker()
    face_rec = FaceRecognizer()

    cap = cv2.VideoCapture(video_path)

    print("Video path:", video_path)

    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    # ===== test đọc frame đầu =====
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read first frame")
        return
    else:
        print("✅ Video read OK")

    # ===== tạo thư mục output =====
    os.makedirs("output", exist_ok=True)

    # ===== đặt tên file KHÔNG bị ghi đè =====
    filename = os.path.basename(video_path)
    name, _ = os.path.splitext(filename)
    save_path = f"output/{name}_result.mp4"

    known_ids = {}     # id -> name
    id_frames = {}     # id -> số frame đã thấy
    unique_ids = set()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    # ===== LOOP VIDEO =====
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ===== scale cho đẹp =====
        h, w = frame.shape[:2]
        scale = w / 640
        thickness = max(2, int(3 * scale))
        font_scale = max(0.6, 0.7 * scale)

        # ===== init writer =====
        if out is None:
            out = cv2.VideoWriter(save_path, fourcc, 20, (w, h))
            print(f"Saving video to: {save_path}")

        objects = tracker.track(frame)

        for (track_id, x1, y1, x2, y2) in objects:

            if track_id == -1:
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            unique_ids.add(track_id)

            # ===== đếm frame =====
            if track_id not in id_frames:
                id_frames[track_id] = 0
            id_frames[track_id] += 1

            # ===== nhận diện ổn định =====
            if track_id not in known_ids:

                if id_frames[track_id] > 5:
                    crop = frame[y1:y2, x1:x2]

                    if crop.size != 0:
                        name = face_rec.recognize(crop)
                    else:
                        name = "Unknown"

                    known_ids[track_id] = name
                else:
                    known_ids[track_id] = "Detecting..."

            name = known_ids[track_id]

            label = f"{name} (ID {track_id})"

            # ===== vẽ =====
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), thickness)

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)

        # ===== tổng =====
        cv2.putText(frame, f"Total: {len(unique_ids)}", (20, int(40*scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)

        # ===== ghi =====
        if out:
            out.write(frame)

        # ===== show =====
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()

    if out:
        out.release()

    cv2.destroyAllWindows()

    print(f"Saved: {save_path}")