"""Audio transcription (Phase 2, POST /transcribe).

Uses faster-whisper on CPU (int8). The model is loaded lazily on the first
request and cached for the process, so the service starts fast and the ~75 MB
(tiny) / ~145 MB (base) weights are only downloaded when audio is first used.
"""

from functools import lru_cache
from typing import Any

from ..config import settings


@lru_cache(maxsize=1)
def _model() -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(
        settings.whisper_model,
        device="cpu",
        compute_type=settings.whisper_compute_type,
    )


def model_name() -> str:
    return settings.whisper_model


def is_loaded() -> bool:
    return _model.cache_info().currsize > 0


def _run(path: str, language: str | None, vad: bool):
    segments, info = _model().transcribe(
        path,
        language=language or None,
        vad_filter=vad,
        beam_size=1,
    )

    # Materialise inside the call: the VAD filter raises while iterating.
    return list(segments), info


def transcribe(path: str, language: str | None = None) -> dict:
    """Return {"text": str, "segments": [{start, end, text}], "language": str}.

    A clip with no speech at all makes the VAD filter raise (max() on an empty
    sequence), so fall back to transcribing without VAD: the caller then gets an
    empty transcript instead of a 502.
    """
    try:
        segments, info = _run(path, language, True)
    except ValueError:
        segments, info = _run(path, language, False)

    rows = []
    parts = []

    for segment in segments:
        text = (segment.text or "").strip()

        if text == "":
            continue

        rows.append({
            "start": round(float(segment.start), 2),
            "end": round(float(segment.end), 2),
            "text": text,
        })
        parts.append(text)

    return {
        "text": " ".join(parts).strip(),
        "segments": rows,
        "language": getattr(info, "language", None),
    }
