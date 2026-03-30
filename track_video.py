import cv2
import os
import time
from src.tracker import Tracker
from src.detector import Detector


def run_video(video_path):
    tracker = Tracker()
    detector = Detector()

    cap = cv2.VideoCapture(video_path)

    print("Video path:", video_path)

    if not cap.isOpened():
        print("Cannot open video")
        return

    os.makedirs("output", exist_ok=True)

    filename = os.path.basename(video_path)
    name, _ = os.path.splitext(filename)
    save_path = f"output/{name}_result.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    prev_time = 0
    smooth_total = 0  # 👉 làm mượt số lượng

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        scale = w / 640
        thickness = max(2, int(3 * scale))
        font_scale = max(0.6, 0.7 * scale)

        if out is None:
            out = cv2.VideoWriter(save_path, fourcc, 20, (w, h))
            print("Saving video to:", save_path)

        # ===== DETECT =====
        detections = detector.predict(frame)

        person_boxes = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det

            # 👉 lọc detection cho ổn định
            if conf < 0.5:
                continue

            person_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        # ===== TRACK =====
        objects = tracker.update(person_boxes)

        active_ids = set()

        for (x1, y1, x2, y2, track_id) in objects:

            if x2 <= x1 or y2 <= y1:
                continue

            active_ids.add(track_id)

            # ===== BOX =====
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)

            # ===== LABEL =====
            label = f"ID {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        # ===== SMOOTH TOTAL =====
        current_total = len(active_ids)

        if current_total > smooth_total:
            smooth_total = current_total
        elif current_total < smooth_total:
            smooth_total -= 1

        # ===== FPS =====
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # ===== UI TEXT =====
        cv2.putText(frame, f"Total: {smooth_total}", (20, int(40 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

        cv2.putText(frame, f"FPS: {int(fps)}", (20, int(80 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)

        # ===== SAVE =====
        if out:
            out.write(frame)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()

    if out:
        out.release()

    cv2.destroyAllWindows()

    print("Saved:", save_path)