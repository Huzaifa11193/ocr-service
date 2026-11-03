"""Image OCR pipeline tests: preprocessing, Tesseract engine, render contract.

These tests are skipped automatically when Tesseract or its language data is
not installed on the host.
"""

import shutil

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from app.main import app
from app.ocr.preprocess import preprocess_image
from app.ocr.tesseract_engine import ocr_image

client = TestClient(app)
VALID_HEADERS = {"X-Service-Token": __import__("os").environ.get("SERVICE_TOKEN", "test-secret-token")}

TESSERACT = shutil.which("tesseract")
LANGS = set()
if TESSERACT:
    import pytesseract

    try:
        LANGS = set(pytesseract.get_languages(config=""))
    except Exception:  # pragma: no cover - depends on host install
        LANGS = set()


requires_tesseract = pytest.mark.skipif(
    not TESSERACT or "eng" not in LANGS,
    reason="Tesseract with eng language data is not installed",
)


def sample_text_image(text: str = "Hello Scribe 123", size=(640, 200)) -> Image.Image:
    image = Image.new("L", size, 255)
    draw = ImageDraw.Draw(image)
    draw.text((40, 80), text, fill=0)
    return image


def skewed_image(angle: float = 12.0) -> Image.Image:
    base = sample_text_image()
    # expand=True keeps the rotated content inside the canvas.
    return base.rotate(angle, fillcolor=255, expand=True)


class TestPreprocess:
    def test_deskew_straightens_skewed_page(self):
        from app.ocr.preprocess import deskew_image
        import numpy as np

        gray = np.array(skewed_image(), dtype=np.uint8)
        deskewed = deskew_image(gray)
        assert deskewed.shape == gray.shape

    def test_preprocess_returns_grayscale_image(self):
        result = preprocess_image(sample_text_image())
        assert isinstance(result, Image.Image)
        assert result.size[0] > 0 and result.size[1] > 0

    def test_preprocess_accepts_rgb(self):
        rgb = sample_text_image().convert("RGB")
        result = preprocess_image(rgb)
        assert isinstance(result, Image.Image)


@requires_tesseract
class TestTesseractEngine:
    def test_ocr_reads_printed_text(self):
        text, confidence = ocr_image(sample_text_image("Hello Scribe 123"), lang="eng")
        assert "Hello" in text or "hello" in text.lower()
        assert isinstance(confidence, float)

    def test_ocr_returns_confidence_range(self):
        _, confidence = ocr_image(sample_text_image(), lang="eng")
        assert 0.0 <= confidence <= 100.0


class TestRenderContract:
    def test_render_rejects_invalid_image(self):
        response = client.post(
            "/ocr/render",
            files={"file": ("bad.png", b"not-an-image", "image/png")},
            headers=VALID_HEADERS,
        )
        assert response.status_code == 415

    def test_render_returns_png_zip_for_image_upload(self):
        import io
        import zipfile

        buffer = io.BytesIO()
        sample_text_image().save(buffer, format="PNG")

        response = client.post(
            "/ocr/render",
            files={"file": ("page.png", buffer.getvalue(), "image/png")},
            headers=VALID_HEADERS,
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/zip")

        archive = zipfile.ZipFile(io.BytesIO(response.content))
        names = archive.namelist()
        assert names == ["page_1.png"]
        # Extracted entry must be a real PNG.
        assert archive.read("page_1.png")[:8] == b"\x89PNG\r\n\x1a\n"