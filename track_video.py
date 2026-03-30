import cv2
import os
from src.tracker import Tracker
from src.face_recognizer import FaceRecognizer
from src.detector import Detector


def run_video(video_path):
    tracker = Tracker()
    detector = Detector()
    face_rec = FaceRecognizer()

    cap = cv2.VideoCapture(video_path)

    print("Video path:", video_path)

    if not cap.isOpened():
        print("Cannot open video")
        return

    # test frame đầu
    ret, frame = cap.read()
    if not ret:
        print("Cannot read first frame")
        return
    else:
        print("Video read OK")

    os.makedirs("output", exist_ok=True)

    filename = os.path.basename(video_path)
    name, _ = os.path.splitext(filename)
    save_path = f"output/{name}_result.mp4"

    known_ids = {}
    id_frames = {}
    unique_ids = set()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

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

        # detect
        detections = detector.predict(frame)

        person_boxes = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            person_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        # tracking
        objects = tracker.update(person_boxes)

        for (x1, y1, x2, y2, track_id) in objects:

            if x2 <= x1 or y2 <= y1:
                continue

            unique_ids.add(track_id)

            if track_id not in id_frames:
                id_frames[track_id] = 0
            id_frames[track_id] += 1

            if track_id not in known_ids or known_ids[track_id] == "Detecting":
                recognized_ids = set()
                if track_id not in recognized_ids and id_frames[track_id] > 5:
                    recognized_ids.add(track_id)

                if id_frames[track_id] > 5:
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    crop = frame[y1:y2, x1:x2]

                    if crop.size != 0:
                        try:
                            name = face_rec.recognize(crop)
                        except:
                            name = "Unknown"
                    else:
                        name = "Unknown"

                    known_ids[track_id] = name
                else:
                    known_ids[track_id] = "Unknown"

            name = known_ids[track_id]
            label = f"ID {track_id}" if name == "" else f"{name} (ID {track_id})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        cv2.putText(frame, f"Total: {len(unique_ids)}", (20, int(40 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

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
    print("Total people:", len(unique_ids))