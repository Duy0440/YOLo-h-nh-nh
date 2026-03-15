import cv2

def load_video(path):
    """Generator: yield frame từ video"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def draw_boxes(frame, dets):
    """Vẽ bbox + class name + confidence rõ, dày hơn"""
    for d in dets:
        x1, y1, x2, y2 = d['xyxy']
        cls, conf = d['name'], d['confidence']

        # Dùng box dày hơn (3) và chữ lớn hơn (0.7)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
        cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return frame