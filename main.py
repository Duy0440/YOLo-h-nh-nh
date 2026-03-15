import argparse, os, cv2
from typing import Optional
from src.detector import Detector
from src.counter import Counter
from src.utils import draw_boxes

def process_video(input_path: str, output_dir: str, model_path: str, conf=0.25, target_class: Optional[str]=None, draw_bboxes=True):
    os.makedirs(output_dir, exist_ok=True)
    det = Detector(model_path, conf)
    cnt = Counter()
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(output_dir, os.path.basename(input_path))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        dets = det.predict(frame)
        if target_class: dets = [d for d in dets if d['name']==target_class]
        cnt.update(dets)
        vis = draw_boxes(frame, dets) if draw_bboxes else frame
        if frame_id % 30 == 0:
            counts = {d['name']: 0 for d in dets}
            for d in dets: counts[d['name']] += 1
            print(f"Frame {frame_id}: total={len(dets)}, " + ", ".join(f"{k}={v}" for k,v in counts.items()))
        writer.write(vis)
        frame_id += 1
    cap.release()
    writer.release()
    print(f"Done video -> {out_path}, total unique objects={cnt.totals()}")

def process_image(input_path: str, output_path: str, model_path: str, conf=0.25, target_class: Optional[str]=None, draw_bboxes=True):
    det = Detector(model_path, conf)
    img = cv2.imread(input_path)
    if img is None: raise IOError(f"Cannot read {input_path}")
    dets = det.predict(img)
    if target_class: dets = [d for d in dets if d['name']==target_class]
    counts = {}
    for d in dets: counts[d['name']] = counts.get(d['name'],0)+1
    print(f"Counts: total={len(dets)}, " + ", ".join(f"{k}={v}" for k,v in counts.items()))
    vis = draw_boxes(img, dets) if draw_bboxes else img
    if draw_bboxes: cv2.putText(vis, f"Count: {len(dets)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.imwrite(output_path, vis)
    print(f"Saved image -> {output_path}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, default='models/yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--target-class', type=str, default=None)
    parser.add_argument('--no-draw', action='store_true', dest='no_draw')
    args = parser.parse_args()

    is_video = os.path.isdir(args.input) or args.input.lower().endswith(('.mp4','.avi','.mkv'))
    if is_video:
        process_video(args.input, args.output, args.model, conf=args.conf, target_class=args.target_class, draw_bboxes=not args.no_draw)
    else:
        process_image(args.input, args.output, args.model, conf=args.conf, target_class=args.target_class, draw_bboxes=not args.no_draw)