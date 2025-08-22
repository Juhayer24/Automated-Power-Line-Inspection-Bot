# src/utils/io.py
import cv2
import os

def video_frame_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def extract_frames(video_path, out_dir, step=30):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            fname = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
        idx += 1
    cap.release()
    return saved
