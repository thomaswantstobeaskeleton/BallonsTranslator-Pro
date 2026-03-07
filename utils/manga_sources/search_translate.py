"""
Translate search query for raw manga sources (NaruRaw, ManhwaRaw, 1kkk).
Raw sites use Japanese/Korean/Chinese titles; this helps by translating an English (or other) query
to the source language before searching. Uses MyMemory free API (no key required).
"""
from __future__ import annotations

from typing import Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore

# MyMemory API: free, no key; langpair: en|ja, en|ko, en|zh-CN
MYMEMORY_URL = "https://api.mymemory.translated.net/get"
TIMEOUT = 10

# Target language code per raw source
RAW_SOURCE_LANG = {
    "naruraw": "ja",   # Japanese
    "manhwaraw": "ko", # Korean
    "onekkk": "zh",    # Chinese (MyMemory uses zh-CN)
}


def translate_search_query(query: str, source_id: str) -> Optional[str]:
    """
    Translate a search query to the raw source language (Japanese, Korean, or Chinese).
    Used when searching NaruRaw, ManhwaRaw, or 1kkk with an English (or other) term.
    Returns translated string, or None if translation fails (caller can fall back to original query).
    """
    if not query or not query.strip():
        return None
    query = query.strip()
    target = RAW_SOURCE_LANG.get(source_id)
    if not target:
        return query
    if target == "zh":
        target = "zh-CN"  # MyMemory
    langpair = f"en|{target}"
    if not requests:
        return None
    try:
        r = requests.get(
            MYMEMORY_URL,
            params={"q": query, "langpair": langpair},
            timeout=TIMEOUT,
            headers={"User-Agent": "BallonsTranslator/1.0"},
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        resp_data = data.get("responseData") or {}
        translated = (resp_data.get("translatedText") or "").strip()
        if translated and translated != query:
            return translated
        return None
    except Exception:
        return None
