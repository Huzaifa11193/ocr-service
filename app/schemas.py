from typing import Literal, Optional

from pydantic import BaseModel, Field

OcrEngine = Literal["tesseract", "vision"]


class PageResult(BaseModel):
    """OCR result for one page of a document."""

    n: int = Field(description="1-based page number")
    text: str = ""
    confidence: Optional[float] = Field(
        default=None,
        description="Average word confidence (0-100); null when the engine does not report one.",
    )
    engine: Optional[str] = Field(
        default=None,
        description="Engine that produced this page (tesseract, vision, text-layer).",
    )


class OcrResponse(BaseModel):
    pages: list[PageResult]
    engine: str


class SinglePageOcrRequest(BaseModel):
    """Form options for POST /ocr/page (sent alongside the image file)."""

    engine: OcrEngine = "tesseract"
    lang: str = ""
    preprocess: bool = True
    handwriting: bool = False


class SinglePageOcrResponse(BaseModel):
    """Result for POST /ocr/page."""

    text: str
    confidence: Optional[float] = None
    engine: str
    language: str
    page: int = 1
