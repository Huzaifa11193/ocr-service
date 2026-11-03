"""Health endpoint tests (Phase 4.1)."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_reports_service_fields():
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["service"] == "ocr"
    assert payload["status"] in ("ok", "degraded")
    assert isinstance(payload["uptime_seconds"], (int, float))
    assert payload["uptime_seconds"] >= 0
    assert payload["started_at"]

    tesseract = payload["tesseract"]
    assert "version" in tesseract
    assert isinstance(tesseract["languages"], list)
    # Phase 2.3: required languages are reported either as present or missing.
    assert set(tesseract["missing_languages"]) <= {"urd", "eng"}