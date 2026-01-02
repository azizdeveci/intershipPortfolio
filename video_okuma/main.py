import json
import matplotlib.pyplot as plt

from video_processor import VideoProcessor
from accuracy import AccuracyEstimator

from ocr_tesseract import TesseractOCR
from ocr_easy import EasyOCR
from ocr_paddle import PaddleOCREngine


def main():
    video = VideoProcessor("video.mp4", sample_every_sec=2)
    frames = video.extract_frames()

    engines = [
        TesseractOCR(),
        EasyOCR(),
        PaddleOCREngine()
    ]

    scores = {e.name: [] for e in engines}

    for frame in frames:
        pre_img = video.preprocess(frame)       # grayscale / threshold
        orig_img = video.read_original(frame)   # BGR (H, W, 3)

        for engine in engines:

            print("ENGINE:", engine.name)
            
            if engine.name == "PaddleOCR":
                text = engine.run(orig_img)
            else:
                 text = engine.run(pre_img)
      # ✅ 2 channel OK

            acc = AccuracyEstimator.estimate(text)
            scores[engine.name].append(acc)

            print(f"[{engine.name}] {frame} → %{acc}")



    with open("ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    # 📊 Visualization
    plt.figure(figsize=(10, 5))
    for name, vals in scores.items():
        plt.plot(vals, marker="o", label=name)

    plt.title("OCR Accuracy Comparison (Video Frames)")
    plt.xlabel("Frame Index")
    plt.ylabel("Accuracy (%)")
    plt.savefig("ocr_comparison.png")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
