from pydantic import BaseModel


class PageResult(BaseModel):
    n: int
    text: str
    confidence: float


class OcrResponse(BaseModel):
    pages: list[PageResult]
    engine: str
