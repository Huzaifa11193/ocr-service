"""POST /ocr/page contract tests (Phase 2.2 / 4.1).

These post a real image over multipart exactly like the Laravel client does.
A previous version of the endpoint declared its options as a ``Form()`` model,
which FastAPI 0.115 does not flatten: every real call returned 422 while the
older auth test still passed because it accepted 415 *or* 422.
"""

import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app
from app.routers import ocr as ocr_router

client = TestClient(app)

VALID_HEADERS = {"X-Service-Token": os.environ.get("SERVICE_TOKEN", "test-secret-token")}


def _png_bytes(text: str = "Scribe page") -> bytes:
    image = Image.new("RGB", (1200, 300), "white")
    ImageDraw.Draw(image).text((40, 60), text, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_page_endpoint_accepts_multipart_options():
    """engine/lang/preprocess must bind as ordinary form fields."""
    response = client.post(
        "/ocr/page",
        data={"engine": "tesseract", "lang": "eng", "preprocess": "0", "handwriting": "0"},
        files={"file": ("page_1.png", b"not-an-image", "image/png")},
        headers=VALID_HEADERS,
    )

    # 415 = form parsed and auth passed, only the image bytes were invalid.
    assert response.status_code == 415, response.text


def test_page_endpoint_rejects_unknown_engine_with_422():
    response = client.post(
        "/ocr/page",
        data={"engine": "magic"},
        files={"file": ("page_1.png", _png_bytes(), "image/png")},
        headers=VALID_HEADERS,
    )

    assert response.status_code == 422
    assert "Unsupported engine" in response.text


def test_page_endpoint_vision_returns_text(monkeypatch):
    """The vision branch must run and return its transcription."""
    seen = {}

    def fake_vision(image):
        seen["size"] = image.size
        return "بہتر شدہ متن\nVision output"

    monkeypatch.setattr(ocr_router, "vision_ocr", fake_vision)

    response = client.post(
        "/ocr/page",
        data={"engine": "vision"},
        files={"file": ("page_1.png", _png_bytes(), "image/png")},
        headers=VALID_HEADERS,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["engine"] == "vision"
    assert "Vision output" in payload["text"]
    # Vision reports no word confidences.
    assert payload["confidence"] is None
    assert seen["size"] == (1200, 300)


def test_page_endpoint_requires_token():
    response = client.post(
        "/ocr/page",
        data={"engine": "vision"},
        files={"file": ("page_1.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 401
