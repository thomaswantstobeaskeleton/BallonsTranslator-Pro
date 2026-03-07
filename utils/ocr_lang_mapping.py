"""
Map source language (e.g. translate_source) to preferred OCR module for ocr_auto_by_language.
"""
from __future__ import annotations

# Language name (as in translate_source dropdown or common names) -> OCR module key
# Use module keys from modules.ocr (e.g. manga_ocr, mit48px, rapidocr, etc.)
OCR_KEY_FOR_LANGUAGE: dict[str, str] = {
    "日本語": "manga_ocr",
    "Japanese": "manga_ocr",
    "한국어": "mit48px",
    "Korean": "mit48px",
    "简体中文": "mit48px",
    "繁體中文": "mit48px",
    "Chinese": "mit48px",
    "English": "mit48px",
}


def get_ocr_key_for_language(lang: str, fallback: str = "mit48px") -> str:
    """Return OCR module key for the given source language name. Uses fallback if no mapping."""
    if not lang or not isinstance(lang, str):
        return fallback
    lang = lang.strip()
    return OCR_KEY_FOR_LANGUAGE.get(lang, fallback)
