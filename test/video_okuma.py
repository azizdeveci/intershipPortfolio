import cv2
import os
import json
import pytesseract
import numpy as np
from PIL import Image
import re
from typing import List


class VideoOCR:
    def __init__(
        self,
        video_path: str,
        output_dir: str = "frames",
        sample_every_sec: int = 2,
        languages: str = "eng+tur"
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.sample_every_sec = sample_every_sec
        self.languages = languages

        os.makedirs(self.output_dir, exist_ok=True)

    # -----------------------------
    # Video → Frame
    # -----------------------------
    def extract_frames(self) -> List[str]:
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * self.sample_every_sec)

        saved_frames = []
        frame_id = 0
        saved_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_interval == 0:
                path = f"{self.output_dir}/frame_{saved_id}.png"
                cv2.imwrite(path, frame)
                saved_frames.append(path)
                saved_id += 1

            frame_id += 1

        cap.release()
        return saved_frames

    # -----------------------------
    # Preprocessing
    # -----------------------------
    def preprocess(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return thresh

    # -----------------------------
    # OCR
    # -----------------------------
    def ocr(self, image: np.ndarray) -> str:
        config = "--oem 3 --psm 6"
        return pytesseract.image_to_string(
            image,
            lang=self.languages,
            config=config
        )

    # -----------------------------
    # Heuristic Accuracy
    # -----------------------------
    def estimate_accuracy(self, text: str) -> float:
        if not text.strip():
            return 0.0

        total_chars = len(text)
        alpha_chars = len(re.findall(r"[a-zA-ZğüşöçıİĞÜŞÖÇ]", text))
        noise_chars = len(re.findall(r"[^a-zA-ZğüşöçıİĞÜŞÖÇ0-9\s]", text))

        alpha_ratio = alpha_chars / max(total_chars, 1)
        noise_penalty = noise_chars / max(total_chars, 1)

        score = alpha_ratio - noise_penalty
        return round(max(0.0, min(score, 1.0)) * 100, 2)

    # -----------------------------
    # Full Pipeline
    # -----------------------------
    def run(self):
        frames = self.extract_frames()
        results = []

        for frame in frames:
            processed = self.preprocess(frame)
            text = self.ocr(processed)
            accuracy = self.estimate_accuracy(text)

            results.append({
                "frame": os.path.basename(frame),
                "text": text.strip(),
                "accuracy_%": accuracy
            })

            print(f"[{frame}] → Accuracy: %{accuracy}")

        with open("ocr_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\n✅ OCR işlemi tamamlandı. Sonuçlar: ocr_results.json")
        return results


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    ocr = VideoOCR(
        video_path="video.mp4",
        sample_every_sec=2,
        languages="eng+tur"
    )
    ocr.run()
