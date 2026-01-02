import easyocr


class EasyOCR:
    name = "EasyOCR"

    def __init__(self, languages=["en", "tr"]):
        self.reader = easyocr.Reader(languages, gpu=False)

    def run(self, image):
        result = self.reader.readtext(image)
        return " ".join([r[1] for r in result])
