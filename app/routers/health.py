import time
from datetime import datetime, timezone

import pytesseract
from fastapi import APIRouter

from ..ocr import whisper_engine

router = APIRouter()

SERVICE_VERSION = "0.3.0"
REQUIRED_LANGUAGES = ("urd", "eng")

_STARTED_AT = time.monotonic()
_STARTED_AT_UTC = datetime.now(timezone.utc)


@router.get("/health")
def health():
    """Report service health, Tesseract version, languages and uptime."""
    tesseract_version: str | None = None
    languages: list[str] = []
    error: str | None = None

    try:
        tesseract_version = str(pytesseract.get_tesseract_version())
    except Exception as exc:  # pragma: no cover - depends on host install
        error = f"Tesseract unavailable: {exc}"

    try:
        languages = sorted(pytesseract.get_languages(config=""))
    except Exception as exc:  # pragma: no cover - depends on host install
        error = error or f"Unable to list Tesseract languages: {exc}"

    missing_languages = [lang for lang in REQUIRED_LANGUAGES if lang not in languages]

    return {
        "status": "ok" if tesseract_version and not missing_languages else "degraded",
        "service": "ocr",
        "version": SERVICE_VERSION,
        "uptime_seconds": round(time.monotonic() - _STARTED_AT, 2),
        "started_at": _STARTED_AT_UTC.isoformat(),
        "tesseract": {
            "version": tesseract_version,
            "languages": languages,
            "missing_languages": missing_languages,
        },
        "whisper": {
            "model": whisper_engine.model_name(),
            "loaded": whisper_engine.is_loaded(),
        },
        "error": error,
    }
