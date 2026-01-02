import re

class AccuracyEstimator:
    @staticmethod
    def estimate(text: str) -> float:
        if not text.strip():
            return 0.0

        total = len(text)
        alpha = len(re.findall(r"[a-zA-ZğüşöçıİĞÜŞÖÇ]", text))
        noise = len(re.findall(r"[^a-zA-ZğüşöçıİĞÜŞÖÇ0-9\s]", text))

        score = (alpha / max(total, 1)) - (noise / max(total, 1))
        return round(max(0, min(score, 1)) * 100, 2)
