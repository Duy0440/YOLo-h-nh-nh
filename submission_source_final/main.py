import os
import sys
import warnings
import logging


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


warnings.filterwarnings("ignore")


logging.getLogger().setLevel(logging.ERROR)


class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass


class DevNullOut:
    def write(self, msg):
        pass
    def flush(self):
        pass

# sys.stdout = DevNullOut()  


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["image", "video"], required=True)
    parser.add_argument("--source", required=True)

    args = parser.parse_args()

    print(f"Running mode: {args.mode}")

    
    if not os.path.exists(args.source):
        print(f"Source not found: {args.source}")
        sys.exit(1)

    if args.mode == "image":
        from detect_image import run_image
        run_image(args.source)

    elif args.mode == "video":
        from track_video import run_video
        run_video(args.source)