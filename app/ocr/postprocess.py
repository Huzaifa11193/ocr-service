import re
import unicodedata

# Arabic characters that have a distinct Urdu form. Applied per character so
# mixed Arabic/Urdu OCR output converges on the Urdu spelling used for search.
CHAR_MAP = {
    "ي": "ی",  # Arabic yeh -> Urdu yeh
    "ى": "ی",  # Arabic alef maksura -> Urdu yeh
    "ك": "ک",  # Arabic kaf -> Urdu kaf
    "ة": "ہ",  # Arabic teh marbuta -> Urdu heh (common word-final form)
}

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def _apply_char_map(text: str) -> str:
    """Replace Arabic character variants with their Urdu equivalents."""
    return "".join(CHAR_MAP.get(char, char) for char in text)


def _remove_zwnj(text: str) -> str:
    """Remove zero-width non-joiners (U+200C) which break search terms."""
    return text.replace("\u200c", "")


def _normalize_whitespace(text: str) -> str:
    """Collapse redundant spaces while keeping paragraph breaks intact."""
    text = _HORIZONTAL_WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()


def postprocess_text(text: str) -> str:
    """Normalize OCR text for storage, search and embedding.

    1. Unicode NFC normalization - canonical representation.
    2. Arabic -> Urdu character mapping.
    3. Zero-width non-joiner removal.
    4. Redundant whitespace cleanup (line breaks are preserved).
    """
    text = unicodedata.normalize("NFC", text or "")
    text = _apply_char_map(text)
    text = _remove_zwnj(text)
    text = _normalize_whitespace(text)
    return text
