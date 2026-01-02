import cv2
import os
import numpy as np


class VideoProcessor:
    def __init__(self, video_path, output_dir="frames", sample_every_sec=2):
        self.video_path = video_path
        self.output_dir = output_dir
        self.sample_every_sec = sample_every_sec
        os.makedirs(output_dir, exist_ok=True)

    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps * self.sample_every_sec)

        frames = []
        i = 0
        saved = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if i % interval == 0:
                path = f"{self.output_dir}/frame_{saved}.png"
                cv2.imwrite(path, frame)
                frames.append(path)
                saved += 1
            i += 1

        cap.release()
        return frames

    @staticmethod
    def preprocess(path):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return th
    
    @staticmethod
    def read_original(path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image could not be read: {path}")
        return img


