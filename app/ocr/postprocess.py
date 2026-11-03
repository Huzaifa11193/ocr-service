import unicodedata
import re
from typing import Iterable

# Mapping of Arabic characters to their Urdu equivalents
CHAR_MAP = {
    "ي": "ی",
    "ك": "ک",
    "ى": "ی",
    "ة": "ہ",
    "دِ": "د",
    "بِ": "ب",
    # Add more mappings as needed
}

def _apply_char_map(text: str) -> str:
    """Replace Arabic characters with Urdu equivalents based on ``CHAR_MAP``.
    The mapping is applied character‑by‑character to keep the function fast.
    """
    return "".join(CHAR_MAP.get(ch, ch) for ch in text)

def _remove_zwnj(text: str) -> str:
    """Zero‑width non‑joiner (U+200C) is removed because it interferes with search.
    """
    return text.replace("\u200c", "")

def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs/newlines into a single space and strip.
    ``re`` is used for speed and readability.
    """
    # Replace any whitespace sequence (including newlines) with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def postprocess_text(text: str) -> str:
    """Perform all post‑processing steps required by the ``arc`` spec.

    1. Unicode NFC normalization – ensures a canonical representation.
    2. Arabic→Urdu character mapping.
    3. Remove zero‑width non‑joiner characters.
    4. Collapse whitespace.

    The function returns a clean string ready for storage, indexing and embedding.
    """
    # Step 1 – canonical composition
    text = unicodedata.normalize("NFC", text)
    # Step 2 – character mapping
    text = _apply_char_map(text)
    # Step 3 – ZWNJ removal
    text = _remove_zwnj(text)
    # Step 4 – whitespace cleanup
    text = _normalize_whitespace(text)
    return text
