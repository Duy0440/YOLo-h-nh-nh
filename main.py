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

# ===== TẮT STDERR =====
class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass

#sys.stderr = DevNull()

# ===== TẮT STDOUT (YOLO hay in ra đây) =====
class DevNullOut:
    def write(self, msg):
        pass
    def flush(self):
        pass

# 👉 QUAN TRỌNG: bật/tắt tùy lúc
# Nếu muốn debug thì comment dòng này
#sys.stdout = DevNullOut()


# ===== MAIN =====
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["image", "video"], required=True)
    parser.add_argument("--source", required=True)

    args = parser.parse_args()

    # 👉 in lại cái cần thiết thôi
    print = sys.__stdout__.write
    print(f"Running mode: {args.mode}\n")

    if args.mode == "image":
        from detect_image import run_image
        run_image(args.source)

    elif args.mode == "video":
        from track_video import run_video
        run_video(args.source)