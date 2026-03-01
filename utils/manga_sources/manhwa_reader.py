"""
Manhwa-Reader Vercel API client (https://manhwa-reader.vercel.app).
Unofficial API: list titles, info+chapters, and chapter image URLs for manhwa/webtoon.
Docs: https://manhwa-reader.vercel.app/api/
"""
from __future__ import annotations

import os
import os.path as osp
import time
from pathlib import Path
from typing import Any, List, Optional

import requests

from utils.logger import logger as LOGGER

BASE_URL = "https://manhwa-reader.vercel.app"
USER_AGENT = "BallonsTranslator/1.0 (https://github.com/dmMaze/BallonsTranslator)"

# Chapter id format for download: "manga_slug|chapter_slug" (e.g. "secret-class|read-secret-class-chapter-1")
MANHWA_READER_ID_SEP = "|"


def _norm_slug(s: str) -> str:
    """Normalize slug for building chapter slug (e.g. strip manga- prefix)."""
    s = (s or "").strip().strip("/")
    if s.startswith("manga-"):
        s = s[6:]
    return s or ""


class ManhwaReaderClient:
    """Client for Manhwa-Reader Vercel API. Search via /api/all, feed via /api/info, download via /api/chapter."""

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

    def is_available(self, timeout_override: Optional[int] = None) -> bool:
        """Return True if the API is up (HTTP 200). Use a short timeout for startup check."""
        try:
            t = timeout_override if timeout_override is not None else self.timeout
            r = self.session.get(f"{BASE_URL}/api/all", timeout=min(t, 8))
            return r.status_code == 200
        except Exception:
            return False

    def search(self, title: str, limit: int = 20) -> List[dict]:
        """
        Search by fetching GET /api/all and filtering by title substring.
        Returns list of dicts with keys: id (slug), title.
        """
        if not title or not title.strip():
            return []
        query = title.strip().lower()
        try:
            r = self._get(f"{BASE_URL}/api/all")
            r.raise_for_status()
            data = r.json()
            # Accept array or { data: [...] } or { manga: [...] }
            if isinstance(data, list):
                items = data
            else:
                items = data.get("data") or data.get("manga") or data.get("list") or []
            out = []
            for item in items[: limit * 3]:  # fetch extra then filter
                if isinstance(item, str):
                    continue
                slug = item.get("slug") or item.get("id") or ""
                tit = item.get("title") or item.get("name") or ""
                if not slug:
                    continue
                if query in (tit or "").lower() or query in (slug or "").lower():
                    out.append({"id": slug, "title": tit or slug, "description": ""})
                    if len(out) >= limit:
                        break
            return out[:limit]
        except requests.RequestException as e:
            # 5xx = server-side; log at DEBUG to avoid flooding when the service is down
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status and 500 <= status < 600:
                LOGGER.debug(f"Manhwa-Reader search failed (server error {status}): {e}")
            else:
                LOGGER.warning(f"Manhwa-Reader search failed: {e}")
            return []
        except Exception as e:
            LOGGER.warning(f"Manhwa-Reader search failed: {e}")
            return []

    def get_feed(
        self,
        manga_id: str,
        translated_language: str = "en",
        limit: int = 500,
        order: str = "asc",
    ) -> List[dict]:
        """
        GET /api/info/:slug. Returns list of chapters with id = manga_slug|chapter_slug.
        translated_language and order are ignored.
        """
        if not manga_id or not str(manga_id).strip():
            return []
        slug = str(manga_id).strip()
        try:
            r = self._get(f"{BASE_URL}/api/info/{slug}")
            r.raise_for_status()
            data = r.json()
            # Chapters might be in data.chapters or data.data.chapters or list at key "chapters"
            chapters_raw = data.get("chapters") or []
            if not chapters_raw and isinstance(data.get("data"), dict):
                chapters_raw = data["data"].get("chapters") or []
            out = []
            norm = _norm_slug(slug)
            for i, ch in enumerate(chapters_raw):
                if isinstance(ch, str):
                    ch_slug = ch
                    ch_num = str(i + 1)
                else:
                    ch_slug = ch.get("slug") or ch.get("chapterSlug") or ch.get("id")
                    ch_num = str(ch.get("chapter") or ch.get("num") or ch.get("number") or (i + 1))
                    if not ch_slug and norm:
                        ch_slug = f"read-{norm}-chapter-{ch_num}"
                if not ch_slug:
                    continue
                composite_id = f"{slug}{MANHWA_READER_ID_SEP}{ch_slug}"
                display = f"Ch.{ch_num}"
                out.append({
                    "id": composite_id,
                    "chapter": ch_num,
                    "volume": None,
                    "title": (ch.get("title") or "") if isinstance(ch, dict) else "",
                    "display": display,
                })
            if order != "asc":
                out.reverse()
            return out[:limit]
        except requests.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status and 500 <= status < 600:
                LOGGER.debug(f"Manhwa-Reader feed failed (server error {status}): {e}")
            else:
                LOGGER.warning(f"Manhwa-Reader feed failed: {e}")
            return []
        except Exception as e:
            LOGGER.warning(f"Manhwa-Reader feed failed: {e}")
            return []

    def get_chapter_images(self, chapter_slug: str) -> Optional[List[str]]:
        """GET /api/chapter/:slug. Returns list of image URLs or None."""
        if not chapter_slug or not chapter_slug.strip():
            return None
        slug = chapter_slug.strip()
        try:
            r = self._get(f"{BASE_URL}/api/chapter/{slug}")
            r.raise_for_status()
            data = r.json()
            urls = data.get("images") or data.get("imageUrls") or data.get("pages") or data.get("data")
            if isinstance(urls, list) and urls:
                return [u if isinstance(u, str) else u.get("url") or u.get("src") for u in urls if u]
            return None
        except requests.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status and 500 <= status < 600:
                LOGGER.debug(f"Manhwa-Reader chapter images failed (server error {status}): {e}")
            else:
                LOGGER.warning(f"Manhwa-Reader chapter images failed: {e}")
            return None
        except Exception as e:
            LOGGER.warning(f"Manhwa-Reader chapter images failed: {e}")
            return None

    def download_chapter(
        self,
        chapter_id: str,
        save_dir: str,
        data_saver: bool = False,
        on_progress: Optional[Any] = None,
    ) -> Optional[str]:
        """
        chapter_id must be manga_slug|chapter_slug. GET /api/chapter/:chapter_slug, download images.
        Saves as 001.ext, 002.ext, ... Returns save_dir on success.
        """
        if not chapter_id or MANHWA_READER_ID_SEP not in chapter_id:
            return None
        parts = chapter_id.split(MANHWA_READER_ID_SEP, 1)
        chapter_slug = parts[1].strip()
        if not chapter_slug:
            return None
        urls = self.get_chapter_images(chapter_slug)
        if not urls:
            return None
        os.makedirs(save_dir, exist_ok=True)
        total = len(urls)
        for i, url in enumerate(urls):
            try:
                r = self._get(url)
                r.raise_for_status()
                ext = Path(url).suffix or ".jpg"
                if not ext.startswith("."):
                    ext = "." + ext
                page_name = f"{i + 1:03d}{ext}"
                path = osp.join(save_dir, page_name)
                with open(path, "wb") as f:
                    f.write(r.content)
                if on_progress:
                    on_progress(i + 1, total, page_name)
            except Exception as e:
                LOGGER.warning(f"Manhwa-Reader download page {i + 1} failed: {e}")
                return None
        return save_dir
