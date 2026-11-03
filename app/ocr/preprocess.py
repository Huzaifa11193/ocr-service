import cv2
import numpy as np
from PIL import Image


def deskew_image(gray: np.ndarray) -> np.ndarray:
    """Detect and correct slight skew in a grayscale image.
    Uses the minimum area rectangle around non‑zero pixels to estimate angle.
    Returns a deskewed grayscale ndarray.
    """
    # Invert colors to get foreground as white for contour detection
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] == 0:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_image(image: Image.Image) -> Image.Image:
    """Grayscale → Deskew → Denoise → Adaptive threshold.
    Returns a binary PIL Image ready for OCR.
    """
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Deskew step
    gray = deskew_image(gray)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # Adaptive threshold
    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    return Image.fromarray(thresholded)
