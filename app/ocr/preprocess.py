import cv2
import numpy as np
from PIL import Image


def deskew_image(gray: np.ndarray) -> np.ndarray:
    """Detect and correct slight skew in a grayscale image.

    Uses the minimum area rectangle around non-zero (foreground) pixels to
    estimate the rotation angle, then rotates the page before thresholding.
    Returns a deskewed grayscale ndarray.
    """
    # Invert colors to get foreground as white for contour detection
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] == 0:
        return gray

    # OpenCV 4.5+ reports the angle in (0, 90]; normalise it to (-45, 45] so
    # upright pages are not rotated by ~90 degrees.
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90

    # Ignore tiny noise and large angles (usually layout, not page skew).
    if abs(angle) < 0.1 or abs(angle) > 15:
        return gray

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


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
