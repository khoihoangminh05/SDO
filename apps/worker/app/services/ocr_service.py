import easyocr
import numpy as np

class OCRService:
    def __init__(self):
        self.reader = None

    def initialize(self):
        # We only need english numbers for traffic lights
        self.reader = easyocr.Reader(['en'], gpu=False)

    def read_text(self, image_crop: np.ndarray) -> str:
        if self.reader is None:
            return ""
        
        # Read text from the cropped image
        results = self.reader.readtext(image_crop)
        text = " ".join([res[1] for res in results])
        return text.strip()

ocr_service = OCRService()
