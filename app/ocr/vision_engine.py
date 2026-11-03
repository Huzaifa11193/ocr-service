import base64
import httpx
from pydantic_settings import BaseSettings

from ..config import settings

class VisionSettings(BaseSettings):
    ollama_url: str = settings.ollama_url
    vision_model: str = settings.ollama_vision_model
    timeout: int = 300  # seconds

    class Config:
        env_prefix = ""
        extra = "ignore"

vision_settings = VisionSettings()

def _encode_image(image_bytes: bytes) -> str:
    """Base64‑encode image bytes for Ollama JSON payload."""
    return base64.b64encode(image_bytes).decode("utf-8")

def vision_ocr(image_bytes: bytes) -> str:
    """Call Ollama Vision model to transcribe the image.

    Prompt (from ``05-OLLAMA-AI-SETUP.md``):
    "Extract all text from this image. Preserve line breaks. Reply only with the extracted text."
    """
    payload = {
        "model": vision_settings.vision_model,
        "prompt": "Extract all text from this image. Preserve line breaks. Reply only with the extracted text.",
        "images": [_encode_image(image_bytes)],
        "stream": False,
    }
    try:
        resp = httpx.post(
            f"{vision_settings.ollama_url}/api/generate",
            json=payload,
            timeout=vision_settings.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as exc:
        raise RuntimeError(f"Vision OCR request failed: {exc}")
