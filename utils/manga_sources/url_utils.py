"""
URL helpers for site-specific manga sources (manhua-translator port).
Used by MangaForFree and ToonGod clients for path building and parsing.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse


def infer_id(value: str) -> str:
    """Extract last path segment from URL, or return value as-is if not a URL."""
    if value.startswith("http://") or value.startswith("https://"):
        path = urlparse(value).path.rstrip("/")
        return path.split("/")[-1] or value
    return value


def infer_url(
    base_url: str,
    value: str,
    kind: str,
    manga_id: str | None = None,
) -> str:
    """Build full URL for manga or chapter. Uses 'manga' path for mangaforfree.com else 'webtoon'."""
    if value.startswith("http://") or value.startswith("https://"):
        return value
    path = "manga" if "mangaforfree.com" in base_url else "webtoon"
    if kind == "manga":
        return f"{base_url.rstrip('/')}/{path}/{value}"
    if kind == "chapter":
        if not manga_id:
            raise ValueError("chapter requires manga_id")
        return f"{base_url.rstrip('/')}/{path}/{manga_id}/{value}/"
    raise ValueError(f"unknown kind: {kind}")


def normalize_base_url(value: str) -> str:
    """Return scheme + netloc from URL, or value stripped of trailing slash."""
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return value.rstrip("/")


def slugify_keyword(keyword: str) -> str:
    """Normalize search keyword to URL slug (lowercase, spaces to hyphens)."""
    value = keyword.strip().lower().replace("_", " ")
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[\s\-]+", "-", value)
    return value.strip("-")


def parse_chapter_range(value: str) -> tuple[int, int]:
    """Parse '1-10' or '1:10' into (start, end). Swaps if start > end."""
    match = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", value)
    if not match:
        raise ValueError("Chapter range format should be e.g. 1-10")
    start = int(match.group(1))
    end = int(match.group(2))
    if start <= 0 or end <= 0:
        raise ValueError("Chapter range must be positive integers")
    if start > end:
        start, end = end, start
    return start, end
