import io

import pytesseract
from PIL import Image


def ocr_image(image: Image.Image, lang: str = "urd+eng", handwriting: bool = False) -> tuple[str, float]:
    # Sparse-text mode handles isolated handwritten words better than the
    # uniform-block default.
    psm = "--psm 11" if handwriting else "--psm 6"
    cfg = f"--oem 1 {psm}"
    text = pytesseract.image_to_string(image, lang=lang, config=cfg)
    data = pytesseract.image_to_data(image, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    words = [(t, int(c)) for t, c in zip(data["text"], data["conf"]) if t.strip() and int(c) >= 0]
    conf = sum(c for _, c in words) / max(len(words), 1) if words else 0.0
    return text, conf
