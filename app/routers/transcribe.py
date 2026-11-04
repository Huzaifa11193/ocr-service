"""POST /transcribe — audio to text (Phase 2 API table in 01-PROJECT-STRUCTURE.md)."""

import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..auth import verify_token
from ..config import settings
from ..ocr import whisper_engine

router = APIRouter(dependencies=[Depends(verify_token)])

ALLOWED_SUFFIXES = (".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".mp4", ".aac")


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(""),
):
    """Transcribe an audio file and return `{text, segments, language}`."""
    suffix = os.path.splitext(file.filename or "")[1].lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format '{suffix or 'unknown'}'. Allowed: {', '.join(ALLOWED_SUFFIXES)}",
        )

    payload = await file.read()
    limit = settings.max_audio_mb * 1024 * 1024

    if len(payload) == 0:
        raise HTTPException(status_code=422, detail="The uploaded audio file is empty.")

    if len(payload) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Audio is larger than the {settings.max_audio_mb} MB limit.",
        )

    handle, path = tempfile.mkstemp(suffix=suffix)

    try:
        with os.fdopen(handle, "wb") as buffer:
            buffer.write(payload)

        try:
            result = whisper_engine.transcribe(path, language.strip() or None)
        except Exception as exc:  # model download, decoding or inference failure
            raise HTTPException(status_code=502, detail=f"Transcription failed: {exc}") from exc
    finally:
        if os.path.exists(path):
            os.unlink(path)

    return {
        "text": result["text"],
        "segments": result["segments"],
        "language": result["language"],
        "model": whisper_engine.model_name(),
    }
