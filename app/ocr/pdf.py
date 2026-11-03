import io
from typing import List, Tuple

from pdf2image import convert_from_bytes
from pydantic import BaseModel

# Optional: use pypdf to check for text layer
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def _has_text_layer(pdf_bytes: bytes) -> bool:
    """Return True if any page of the PDF contains extractable text.
    Uses ``pypdf`` to iterate pages and check for non‑empty ``extract_text``.
    """
    if not PdfReader:
        return False
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            if page.extract_text():
                return True
    except Exception:
        pass
    return False


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract raw text from PDFs that already have a text layer.
    Returns a single concatenated string.
    """
    if not PdfReader:
        raise RuntimeError("pypdf not installed for text‑layer extraction")
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts: List[str] = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            texts.append(txt)
    return "\n".join(texts)


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[bytes]:
    """Render each PDF page to a PNG image (as bytes).
    ``pdf2image`` returns ``PIL.Image`` objects – we convert them to PNG bytes.
    """
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    out: List[bytes] = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


class PdfResult(BaseModel):
    has_text_layer: bool
    extracted_text: str | None = None  # present only if has_text_layer=True
    images: List[bytes] | None = None  # PNG bytes of each rendered page when OCR needed
