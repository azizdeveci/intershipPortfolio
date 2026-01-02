import pytesseract


class TesseractOCR:
    name = "Tesseract"

    def __init__(self, lang="eng+tur"):
        self.lang = lang

    def run(self, image):
        return pytesseract.image_to_string(
            image,
            lang=self.lang,
            config="--oem 3 --psm 6"
        )
