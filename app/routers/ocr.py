import io

from fastapi import APIRouter, Depends, File, Form, UploadFile
from PIL import Image

from ..auth import verify_token
from ..ocr.preprocess import preprocess_image
from ..ocr.tesseract_engine import ocr_image
from ..schemas import OcrResponse, PageResult

router = APIRouter(dependencies=[Depends(verify_token)])

try:
    from pdf2image import convert_from_bytes

    PDF_SUPPORT = True
except Exception:
    PDF_SUPPORT = False


@router.post("/ocr", response_model=OcrResponse)
async def ocr(
    file: UploadFile = File(...),
    lang: str = Form("urd+eng"),
    engine: str = Form("tesseract"),
    preprocess: str = Form("1"),
    handwriting: str = Form("0"),
):
    content = await file.read()
    pages: list[PageResult] = []

    images: list[Image.Image] = []
    mime = file.content_type or ""
    if "pdf" in mime and PDF_SUPPORT:
        try:
            images = convert_from_bytes(content, dpi=300)
        except Exception:
            images = []
    if not images:
        try:
            images = [Image.open(io.BytesIO(content))]
        except Exception:
            images = []

    if not images:
        return OcrResponse(pages=[PageResult(n=1, text="", confidence=0.0)], engine=engine)

    use_preprocess = preprocess.lower() not in ("0", "false", "off", "")
    handwriting_mode = handwriting.lower() not in ("0", "false", "off", "")

    for i, img in enumerate(images, start=1):
        if use_preprocess:
            try:
                img = preprocess_image(img)
            except Exception:
                pass
        text, conf = ocr_image(img, lang=lang, handwriting=handwriting_mode)
        pages.append(PageResult(n=i, text=text, confidence=round(conf, 2)))

    return OcrResponse(pages=pages, engine=engine)
