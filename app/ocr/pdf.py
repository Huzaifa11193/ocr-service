import io
from dataclasses import dataclass
from typing import List, Optional

from pdf2image import convert_from_bytes
from PIL import Image

try:  # pypdf is optional at import time so the service still boots without it.
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - exercised only on broken installs
    PdfReader = None

# A page needs at least this many characters to count as having a text layer.
# Shorter strings are usually page numbers or scanner artefacts, not real text.
MIN_PAGE_TEXT_CHARS = 20

# At least this share of pages must contain text for the PDF to be treated as
# digital. Hybrid PDFs (mostly scans with a few digital pages) still go to OCR.
MIN_TEXT_PAGE_RATIO = 0.5


class PdfError(RuntimeError):
    """Raised when a PDF cannot be read or rendered."""


@dataclass
class PdfAnalysis:
    """Result of inspecting a PDF once.

    ``pages`` is always populated with the extracted text for every page (an
    empty string when the page has no text layer). ``images`` is only rendered
    when the PDF is scanned, so digital PDFs never pay the rendering cost.
    """

    has_text_layer: bool
    pages: List[str]
    images: Optional[List[Image.Image]] = None

    @property
    def text(self) -> str:
        return "\n".join(page for page in self.pages if page.strip())


def _read_pages_text(pdf_bytes: bytes) -> List[str]:
    if PdfReader is None:
        return []

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception:
        return []


def _is_digital(pages: List[str]) -> bool:
    if not pages:
        return False

    text_pages = sum(1 for page in pages if len(page.strip()) >= MIN_PAGE_TEXT_CHARS)
    return text_pages > 0 and (text_pages / len(pages)) >= MIN_TEXT_PAGE_RATIO


def has_text_layer(pdf_bytes: bytes) -> bool:
    """Return True when the PDF has a usable, extractable text layer."""
    return _is_digital(_read_pages_text(pdf_bytes))


def extract_text_pages(pdf_bytes: bytes) -> List[str]:
    """Extract text page-by-page from a digital PDF."""
    if PdfReader is None:
        raise PdfError("pypdf is not installed; cannot read the PDF text layer")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:
        raise PdfError(f"Unable to extract PDF text: {exc}") from exc


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
    """Render every PDF page to a PIL image at ``dpi`` for OCR."""
    try:
        return convert_from_bytes(pdf_bytes, dpi=dpi)
    except Exception as exc:
        raise PdfError(f"Unable to render PDF pages: {exc}") from exc


def analyze_pdf(pdf_bytes: bytes, dpi: int = 300) -> PdfAnalysis:
    """Inspect a PDF and return text for digital files or images for scans.

    Digital PDFs bypass OCR entirely. Scanned PDFs are rendered to images at
    ``dpi`` only when no clean text layer exists.
    """
    pages = _read_pages_text(pdf_bytes)
    if _is_digital(pages):
        return PdfAnalysis(has_text_layer=True, pages=pages)

    images = pdf_to_images(pdf_bytes, dpi=dpi)
    return PdfAnalysis(has_text_layer=False, pages=pages, images=images)
