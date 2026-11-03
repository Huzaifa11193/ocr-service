import base64
import io

import httpx
from PIL import Image, ImageOps

from ..config import settings

VISION_PROMPT = (
    "Transcribe all text in this image exactly. Preserve line breaks. "
    "Languages: Urdu and English. Output only the text."
)

# Long edge bounds from 05-OLLAMA-AI-SETUP.md: smaller images stay legible
# while large scans are downscaled to keep CPU memory and latency reasonable.
MIN_LONG_EDGE = 1280
MAX_LONG_EDGE = 1600
MAX_UPSCALE = 2.0


def _resize_for_vision(image: Image.Image) -> Image.Image:
    """Fit the image long edge into the 1280-1600 px range."""
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    long_edge = max(width, height)
    if long_edge == 0:
        return image

    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
    elif long_edge < MIN_LONG_EDGE:
        scale = min(MIN_LONG_EDGE / long_edge, MAX_UPSCALE)
    else:
        return image

    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _prepare_image(image: Image.Image | bytes) -> bytes:
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    else:
        image = image.copy()

    prepared = _resize_for_vision(image)
    buffer = io.BytesIO()
    prepared.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _encode_image(image_bytes: bytes) -> str:
    """Base64-encode image bytes for the Ollama JSON payload."""
    return base64.b64encode(image_bytes).decode("utf-8")


def vision_ocr(image: Image.Image | bytes) -> str:
    """Transcribe a single page with the Ollama vision model.

    ``image`` may be a PIL image (from a PDF render) or raw image bytes. The
    image is resized before upload; the prompt matches 05-OLLAMA-AI-SETUP.md.
    """
    image_bytes = _prepare_image(image)
    payload = {
        "model": settings.ollama_vision_model,
        "prompt": VISION_PROMPT,
        "images": [_encode_image(image_bytes)],
        "stream": False,
        "options": {"temperature": 0},
    }

    try:
        response = httpx.post(
            f"{settings.ollama_url}/api/generate",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"Vision OCR request failed: {exc}") from exc

    return (data.get("response") or "").strip()
