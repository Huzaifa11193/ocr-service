"""POST /transcribe tests (Phase 2 API table).

The model itself is replaced here so the suite stays fast and offline; the
real whisper path is covered by test_transcribe_live below (opt-in) and by a
manual run against spoken audio.
"""

import io
import os
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ocr import whisper_engine

client = TestClient(app)

VALID_HEADERS = {"X-Service-Token": os.environ.get("SERVICE_TOKEN", "test-secret-token")}


def _wav_bytes(seconds: float = 0.5) -> bytes:
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))

    return buffer.getvalue()


def test_transcribe_requires_token():
    response = client.post(
        "/transcribe",
        files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 401


def test_transcribe_rejects_unsupported_format():
    response = client.post(
        "/transcribe",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        headers=VALID_HEADERS,
    )

    assert response.status_code == 415
    assert "Unsupported audio format" in response.text


def test_transcribe_rejects_empty_audio():
    response = client.post(
        "/transcribe",
        files={"file": ("clip.wav", b"", "audio/wav")},
        headers=VALID_HEADERS,
    )

    assert response.status_code == 422


def test_transcribe_returns_text_and_segments(monkeypatch):
    seen = {}

    def fake_transcribe(path, language=None):
        seen["path"] = path
        seen["language"] = language
        seen["exists"] = os.path.exists(path)

        return {
            "text": "یہ آواز کا متن ہے۔",
            "segments": [{"start": 0.0, "end": 1.5, "text": "یہ آواز کا متن ہے۔"}],
            "language": "ur",
        }

    monkeypatch.setattr(whisper_engine, "transcribe", fake_transcribe)

    response = client.post(
        "/transcribe",
        files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        data={"language": "ur"},
        headers=VALID_HEADERS,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["text"] == "یہ آواز کا متن ہے۔"
    assert payload["segments"][0]["end"] == 1.5
    assert payload["language"] == "ur"
    assert payload["model"] == whisper_engine.model_name()
    assert seen["language"] == "ur"
    # The temp file is cleaned up after transcription.
    assert seen["exists"] is True
    assert not os.path.exists(seen["path"])


def test_transcribe_reports_model_failures_as_502(monkeypatch):
    def boom(path, language=None):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(whisper_engine, "transcribe", boom)

    response = client.post(
        "/transcribe",
        files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        headers=VALID_HEADERS,
    )

    assert response.status_code == 502
    assert "Transcription failed" in response.text


@pytest.mark.skipif(
    os.environ.get("WHISPER_LIVE") != "1",
    reason="Set WHISPER_LIVE=1 to download and run the real whisper model.",
)
def test_transcribe_live_with_real_model():
    """Opt-in: runs faster-whisper end to end on a silent clip."""
    response = client.post(
        "/transcribe",
        files={"file": ("clip.wav", _wav_bytes(1.0), "audio/wav")},
        headers=VALID_HEADERS,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload["text"], str)
    assert isinstance(payload["segments"], list)
    assert payload["model"] == whisper_engine.model_name()
