"""
image_processor.py
------------------
Processes uploaded image files.
- Runs OCR (PaddleOCR) to extract text
- Describes image intent based on extracted text
- Returns structured dict for LLM
"""

from __future__ import annotations
import os
from pathlib import Path


def process_image(image_path: str) -> dict:
    """
    Extract text from an image via OCR and return
    a structured dict ready for cleaning + LLM.

    Args:
        image_path: absolute or relative path to the uploaded image

    Returns:
        dict with ocr_text, title, source_type
    """
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    try:
        from utils.ocr import extract_text
        ocr_text = extract_text(image_path)
    except Exception as e:
        ocr_text = ""
        print(f"OCR failed: {e}")

    filename = Path(image_path).name
    title    = f"Image: {filename}"

    # If OCR returned nothing useful, note it
    if not ocr_text or len(ocr_text.strip()) < 5:
        ocr_text = "[No readable text found in image]"

    return {
        "source_type": "local_image",
        "title":       title,
        "ocr_text":    ocr_text,
    }
