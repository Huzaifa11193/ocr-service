"""OCR image handling and text post-processing normalization tests (Phase 4.1)."""

import unicodedata

from app.ocr.postprocess import postprocess_text


class TestPostprocess:
    def test_nfc_normalization(self):
        # Decomposed "a" + combining acute -> composed form.
        decomposed = "á"
        result = postprocess_text(decomposed)
        assert result == unicodedata.normalize("NFC", "á")

    def test_arabic_yeh_becomes_urdu_yeh(self):
        # Arabic yeh U+064A must map to Urdu yeh U+06CC.
        assert postprocess_text(" \u0639\u0631\u0628\u064a ") == "\u0639\u0631\u0628\u06cc"

    def test_arabic_alef_maksura_becomes_urdu_yeh(self):
        assert postprocess_text("مصطفى") == "مصطفی"

    def test_arabic_kaf_becomes_urdu_kaf(self):
        assert postprocess_text("كتاب") == "کتاب"

    def test_arabic_teh_marbuta_becomes_urdu_heh(self):
        assert postprocess_text("مدرسة") == "مدرسہ"

    def test_zwnj_is_removed(self):
        assert postprocess_text("م\u200cتن") == "متن"

    def test_redundant_whitespace_collapsed(self):
        assert postprocess_text("  hello   world  ") == "hello world"

    def test_paragraph_breaks_preserved(self):
        assert postprocess_text("line one\n\n\n\nline two") == "line one\n\nline two"

    def test_line_breaks_preserved(self):
        assert postprocess_text("line one\nline two") == "line one\nline two"

    def test_none_like_input_is_safe(self):
        assert postprocess_text("") == ""
        assert postprocess_text(None) == ""


class TestOcrPageContract:
    def test_page_response_schema_defaults(self):
        from app.schemas import SinglePageOcrResponse

        result = SinglePageOcrResponse(text="hello", engine="tesseract", language="urd+eng")
        assert result.engine == "tesseract"
        assert result.language == "urd+eng"
        assert result.confidence is None
        assert result.page == 1

    def test_engine_rejects_unknown_value(self):
        from fastapi import HTTPException
        from app.routers.ocr import _validate_engine

        assert _validate_engine("tesseract") == "tesseract"
        assert _validate_engine("VISION") == "vision"

        with __import__("pytest").raises(HTTPException) as exc:
            _validate_engine("gpt4")
        assert exc.value.status_code == 422