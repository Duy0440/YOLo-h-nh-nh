import cv2

def draw_boxes(frame, detections, objects=None):

    for i, d in enumerate(detections):

        x1, y1, x2, y2 = d["xyxy"]

        label = f"{d['name']} {d['confidence']:.2f}"

        if objects is not None:
            label = f"ID {i} {label}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(
            frame,
            label,
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    return frame