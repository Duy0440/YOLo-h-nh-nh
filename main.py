import os
import sys
import warnings
import logging

# ===== TẮT LOG TensorFlow =====
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ===== TẮT WARNING =====
warnings.filterwarnings("ignore")

# ===== TẮT LOG Python =====
logging.getLogger().setLevel(logging.ERROR)

# ===== TẮT STDERR (tuỳ chọn)
class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass

# sys.stderr = DevNull()  # bật nếu muốn

# ===== TẮT STDOUT (YOLO spam)
class DevNullOut:
    def write(self, msg):
        pass
    def flush(self):
        pass

# sys.stdout = DevNullOut()  # ❌ KHÔNG bật khi debug

# ===== MAIN =====
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["image", "video"], required=True)
    parser.add_argument("--source", required=True)

    args = parser.parse_args()

    # ✅ print bình thường
    print(f"Running mode: {args.mode}")

    if args.mode == "image":
        from detect_image import run_image
        run_image(args.source)

    elif args.mode == "video":
        from track_video import run_video
        run_video(args.source)