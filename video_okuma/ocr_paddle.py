from paddleocr import PaddleOCR
import cv2


class PaddleOCREngine:
    name = "PaddleOCR"

    def __init__(self):
        self.ocr = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=2.0
        )

    def preprocess(self, img):
        h, w = img.shape[:2]
        if max(h, w) < 1200:
            scale = 1200 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def run(self, image):
        image = self.preprocess(image)
        result = self.ocr.ocr(image)

        if not result:
            return ""

        lines = []
        for line in result:
            lines.append(" ".join(w[1][0] for w in line))

        return "\n".join(lines)
