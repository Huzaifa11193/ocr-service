import io
import zipfile
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from ..auth import verify_token
from ..config import settings
from ..ocr.pdf import PdfError, analyze_pdf, pdf_to_images
from ..ocr.postprocess import postprocess_text
from ..ocr.preprocess import preprocess_image
from ..ocr.tesseract_engine import ocr_image
from ..ocr.vision_engine import vision_ocr
from ..schemas import OcrResponse, PageResult, SinglePageOcrRequest, SinglePageOcrResponse

router = APIRouter(dependencies=[Depends(verify_token)])

_ENGINES = {"tesseract", "vision"}
_TEXT_LAYER_ENGINE = "text-layer"


def _validate_engine(engine: str) -> str:
    engine = (engine or "tesseract").strip().lower()
    if engine not in _ENGINES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported engine '{engine}'. Use one of: {', '.join(sorted(_ENGINES))}.",
        )
    return engine


def _as_bool(value: str | bool | None, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() not in ("0", "false", "off", "no", "")


def _is_pdf(content: bytes, content_type: str, filename: str) -> bool:
    if "pdf" in (content_type or "").lower():
        return True
    if (filename or "").lower().endswith(".pdf"):
        return True
    return content[:5] == b"%PDF-"


def _load_image(content: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(content))
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=415, detail="Uploaded file is not a valid image.") from exc


async def _transcribe_image(
    image: Image.Image,
    engine: str,
    lang: str,
    use_preprocess: bool,
    handwriting: bool,
) -> tuple[str, float | None]:
    """Run the requested engine for a single page image."""
    if engine == "vision":
        # Vision models work best on the original page; binarising destroys
        # stroke detail that helps handwriting recognition.
        try:
            text = await run_in_threadpool(vision_ocr, image)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return postprocess_text(text), None

    if use_preprocess:
        try:
            image = await run_in_threadpool(preprocess_image, image)
        except Exception:
            # Preprocessing is best-effort; fall back to the original page.
            pass

    text, confidence = await run_in_threadpool(ocr_image, image, lang, handwriting)
    return postprocess_text(text), round(float(confidence), 2)


@router.post("/ocr", response_model=OcrResponse)
async def ocr(
    file: UploadFile = File(...),
    lang: str = Form(""),
    engine: str = Form("tesseract"),
    preprocess: str = Form("1"),
    handwriting: str = Form("0"),
):
    """OCR a PDF or a single image.

    Digital PDFs with a clean text layer are extracted directly; scanned PDFs
    are rendered to images at ``OCR_DPI`` (300 by default) before OCR.
    """
    engine = _validate_engine(engine)
    language = (lang or settings.tesseract_lang).strip()
    use_preprocess = _as_bool(preprocess)
    handwriting_mode = _as_bool(handwriting, default=False)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or ""
    content_type = file.content_type or ""

    if _is_pdf(content, content_type, filename):
        try:
            analysis = await run_in_threadpool(analyze_pdf, content, settings.ocr_dpi)
        except PdfError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if analysis.has_text_layer:
            pages = [
                PageResult(
                    n=index,
                    text=postprocess_text(page_text),
                    confidence=100.0,
                    engine=_TEXT_LAYER_ENGINE,
                )
                for index, page_text in enumerate(analysis.pages, start=1)
            ]
            return OcrResponse(pages=pages, engine=_TEXT_LAYER_ENGINE)

        images = analysis.images or []
    else:
        images = [_load_image(content)]

    if not images:
        raise HTTPException(
            status_code=422,
            detail="No pages could be read from the uploaded file.",
        )

    pages: list[PageResult] = []
    for index, image in enumerate(images, start=1):
        text, confidence = await _transcribe_image(
            image, engine, language, use_preprocess, handwriting_mode
        )
        pages.append(PageResult(n=index, text=text, confidence=confidence, engine=engine))

    return OcrResponse(pages=pages, engine=engine)


@router.post("/ocr/render")
async def render(
    file: UploadFile = File(...),
    dpi: int = Form(0),
):
    """Render a PDF (or normalise an image) to a ZIP of 300 DPI PNG pages.

    The Laravel job stores these files as ``pages/page_{n}.png`` so the viewer
    and the vision-enhancement job have a stable image per page.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    render_dpi = dpi if 0 < dpi <= 600 else settings.ocr_dpi
    filename = file.filename or ""
    content_type = file.content_type or ""

    if _is_pdf(content, content_type, filename):
        try:
            images = await run_in_threadpool(pdf_to_images, content, render_dpi)
        except PdfError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        images = [_load_image(content)]

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, image in enumerate(images, start=1):
            page = io.BytesIO()
            image.save(page, format="PNG", optimize=True)
            archive.writestr(f"page_{index}.png", page.getvalue())

    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="pages.zip"'},
    )


@router.post("/ocr/page", response_model=SinglePageOcrResponse)
async def ocr_page(
    file: UploadFile = File(...),
    engine: str = Form("tesseract"),
    lang: str = Form(""),
    preprocess: str | None = Form(None),
    handwriting: str | None = Form(None),
):
    """OCR a single page image, used for per-page vision enhancement.

    Fields are declared individually rather than as a ``Form()`` model: with
    FastAPI 0.115 the model form is not flattened, so the service expected a
    single ``options`` field and every real (multipart) call failed with 422.
    """
    engine = _validate_engine(engine)
    options = SinglePageOcrRequest(
        engine=engine,
        lang=lang,
        preprocess=_as_bool(preprocess, default=True),
        handwriting=_as_bool(handwriting, default=False),
    )
    language = (options.lang or settings.tesseract_lang).strip()

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if _is_pdf(content, file.content_type or "", file.filename or ""):
        raise HTTPException(
            status_code=415,
            detail="This endpoint accepts a single image, not a PDF.",
        )

    image = _load_image(content)
    text, confidence = await _transcribe_image(
        image, engine, language, options.preprocess, options.handwriting
    )

    return SinglePageOcrResponse(
        text=text,
        confidence=confidence,
        engine=engine,
        language=language,
    )
