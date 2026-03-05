from paddleocr import PaddleOCR
from PIL import Image
import os


# Initialize OCR once (faster for multiple calls)
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)


def extract_text(image_path: str) -> str:
    """
    Extract text from an image using PaddleOCR
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:

        result = ocr_engine.ocr(image_path)

        extracted_text = []

        for line in result[0]:
            extracted_text.append(line[1][0])

        return "\n".join(extracted_text)

    except Exception as e:
        return f"OCR Error: {str(e)}"