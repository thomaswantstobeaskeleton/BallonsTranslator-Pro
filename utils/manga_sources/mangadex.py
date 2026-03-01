"""
MangaDex API client for searching manga and downloading chapters.
Uses public API: https://api.mangadex.org/docs/
"""
from __future__ import annotations

import os
import os.path as osp
import re
import time
from pathlib import Path
from typing import Any, List, Optional

import requests

from utils.logger import logger as LOGGER

BASE_URL = "https://api.mangadex.org"
USER_AGENT = "BallonsTranslator/1.0 (https://github.com/dmMaze/BallonsTranslator)"


def _sanitize_filename(name: str) -> str:
    """Replace characters that are invalid in filenames."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name.strip(". ") or "unnamed"


def _get_title(attrs: dict) -> str:
    """Get best available title from manga attributes."""
    titles = attrs.get("title") or {}
    return (
        titles.get("en")
        or titles.get("ja")
        or titles.get("ko")
        or next(iter(titles.values()), "Unknown")
    )


def _get_chapter_display(attrs: dict) -> str:
    """Format chapter number and title for display."""
    ch = attrs.get("chapter") or "?"
    vol = attrs.get("volume")
    title = (attrs.get("title") or "").strip()
    parts = [f"Ch.{ch}"]
    if vol:
        parts.append(f"Vol.{vol}")
    if title:
        parts.append(title)
    return " – ".join(parts)


class MangaDexClient:
    def __init__(self, timeout: int = 30, request_delay: float = 0.3):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.timeout = timeout
        self.request_delay = max(0.0, float(request_delay))

    def _throttle(self):
        if self.request_delay > 0:
            time.sleep(self.request_delay)

    def _get(self, url: str, **kwargs) -> requests.Response:
        self._throttle()
        return self.session.get(url, timeout=self.timeout, **kwargs)

    def search(
        self,
        title: str,
        limit: int = 20,
        original_language: Optional[str] = None,
    ) -> List[dict]:
        """
        Search manga by title. Returns list of dicts with keys: id, title.
        If original_language is set (e.g. 'ja', 'ko', 'zh'), only manga with that
        original language are returned (for finding raw/untranslated material).
        """
        if not title.strip():
            return []
        try:
            params = {"title": title.strip(), "limit": limit}
            if original_language:
                params["originalLanguage[]"] = [original_language]
            r = self._get(f"{BASE_URL}/manga", params=params)
            r.raise_for_status()
            data = r.json()
            if data.get("result") != "ok":
                return []
            out = []
            for item in data.get("data", []):
                if item.get("type") != "manga":
                    continue
                attrs = item.get("attributes") or {}
                out.append(
                    {
                        "id": item.get("id"),
                        "title": _get_title(attrs),
                        "description": (attrs.get("description") or {}).get("en", "")[:200],
                    }
                )
            return out
        except Exception as e:
            LOGGER.warning(f"MangaDex search failed: {e}")
            return []

    def get_feed(
        self,
        manga_id: str,
        translated_language: str = "en",
        limit: int = 500,
        order: str = "asc",
    ) -> List[dict]:
        """
        Get chapter feed for a manga. Returns list of dicts with keys: id, chapter, volume, title, display.
        """
        if not manga_id:
            return []
        try:
            r = self._get(
                f"{BASE_URL}/manga/{manga_id}/feed",
                params={
                    "translatedLanguage[]": translated_language,
                    "limit": limit,
                    "order[chapter]": order,
                },
            )
            r.raise_for_status()
            data = r.json()
            if data.get("result") != "ok":
                return []
            out = []
            for item in data.get("data", []):
                if item.get("type") != "chapter":
                    continue
                attrs = item.get("attributes") or {}
                out.append(
                    {
                        "id": item.get("id"),
                        "chapter": attrs.get("chapter"),
                        "volume": attrs.get("volume"),
                        "title": (attrs.get("title") or "").strip(),
                        "display": _get_chapter_display(attrs),
                    }
                )
            return out
        except Exception as e:
            LOGGER.warning(f"MangaDex feed failed: {e}")
            return []

    def get_chapter_by_id(self, chapter_id: str) -> Optional[dict]:
        """
        Fetch a single chapter by ID (e.g. from a MangaDex chapter URL).
        Returns dict with keys: id, chapter, volume, title, display, or None.
        """
        if not chapter_id or not chapter_id.strip():
            return None
        chapter_id = chapter_id.strip()
        try:
            r = self._get(f"{BASE_URL}/chapter/{chapter_id}")
            r.raise_for_status()
            data = r.json()
            if data.get("result") != "ok":
                return None
            item = data.get("data")
            if not item or item.get("type") != "chapter":
                return None
            attrs = item.get("attributes") or {}
            return {
                "id": item.get("id"),
                "chapter": attrs.get("chapter"),
                "volume": attrs.get("volume"),
                "title": (attrs.get("title") or "").strip(),
                "display": _get_chapter_display(attrs),
            }
        except Exception as e:
            LOGGER.warning(f"MangaDex get chapter failed: {e}")
            return None

    def get_chapter_urls(self, chapter_id: str, data_saver: bool = False) -> Optional[dict]:
        """
        Get at-home server info for a chapter. Returns dict with baseUrl, hash, filenames list.
        """
        if not chapter_id:
            return None
        try:
            r = self._get(f"{BASE_URL}/at-home/server/{chapter_id}")
            r.raise_for_status()
            data = r.json()
            base_url = data.get("baseUrl")
            ch = data.get("chapter") or {}
            h = ch.get("hash")
            filenames = ch.get("dataSaver" if data_saver else "data") or []
            if not base_url or not h or not filenames:
                return None
            return {
                "baseUrl": base_url.rstrip("/"),
                "hash": h,
                "filenames": filenames,
                "quality": "data-saver" if data_saver else "data",
            }
        except Exception as e:
            LOGGER.warning(f"MangaDex at-home failed for {chapter_id}: {e}")
            return None

    def download_chapter(
        self,
        chapter_id: str,
        save_dir: str,
        data_saver: bool = False,
        on_progress: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Download a chapter's images to save_dir. Saves pages as 001.ext, 002.ext, ...
        so BallonsTranslator loads them in order. Returns save_dir on success, None on failure.
        on_progress(current, total, filename) is called for each image if provided.
        """
        info = self.get_chapter_urls(chapter_id, data_saver=data_saver)
        if not info:
            return None
        os.makedirs(save_dir, exist_ok=True)
        base = info["baseUrl"]
        quality = info["quality"]
        h = info["hash"]
        filenames = info["filenames"]
        total = len(filenames)
        for i, fn in enumerate(filenames):
            url = f"{base}/{quality}/{h}/{fn}"
            try:
                r = self._get(url)
                r.raise_for_status()
                ext = Path(fn).suffix or ".png"
                page_name = f"{i + 1:03d}{ext}"
                path = osp.join(save_dir, page_name)
                with open(path, "wb") as f:
                    f.write(r.content)
                if on_progress:
                    on_progress(i + 1, total, page_name)
            except Exception as e:
                LOGGER.warning(f"Failed to download page {i + 1}: {e}")
                return None
        return save_dir
