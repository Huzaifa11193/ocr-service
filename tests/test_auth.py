"""Token authentication tests for the OCR service (Phase 4.1)."""

import asyncio
import os

import pytest
from fastapi.testclient import TestClient

from app.auth import verify_token  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)

VALID_HEADERS = {"X-Service-Token": os.environ.get("SERVICE_TOKEN", "test-secret-token")}


def test_health_requires_no_token():
    response = client.get("/health")
    assert response.status_code == 200


def test_ocr_rejects_request_without_token():
    response = client.post("/ocr", files={"file": ("a.png", b"\x89PNG", "image/png")})
    assert response.status_code == 401


def test_ocr_rejects_wrong_token():
    response = client.post(
        "/ocr",
        files={"file": ("a.png", b"\x89PNG", "image/png")},
        headers={"X-Service-Token": "wrong"},
    )
    assert response.status_code == 401


def test_ocr_page_accepts_valid_token():
    # Auth passes before payload validation, so 415/422 proves the token worked.
    response = client.post(
        "/ocr/page",
        data={"engine": "tesseract"},
        files={"file": ("a.png", b"not-an-image", "image/png")},
        headers=VALID_HEADERS,
    )
    assert response.status_code in (415, 422)


def test_verify_token_direct():
    from fastapi import HTTPException

    token = VALID_HEADERS["X-Service-Token"]
    assert asyncio.run(verify_token(token)) is True

    with pytest.raises(HTTPException) as exc:
        asyncio.run(verify_token("wrong"))
    assert exc.value.status_code == 401