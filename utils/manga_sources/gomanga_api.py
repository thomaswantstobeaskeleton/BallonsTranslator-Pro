"""
GOMANGA-API client (https://gomanga-api.vercel.app).
Unofficial API; provides search, chapter list, and direct chapter image URLs for download.
Docs: https://gomanga-api.vercel.app/docs/introduction
"""
from __future__ import annotations

import os
import os.path as osp
import time
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import quote

import requests

from utils.logger import logger as LOGGER

BASE_URL = "https://gomanga-api.vercel.app"
USER_AGENT = "BallonsTranslator/1.0 (https://github.com/dmMaze/BallonsTranslator)"

# Chapter id format for download: "manga_slug|chapter_num" (e.g. "solo-leveling|1")
GOMANGA_ID_SEP = "|"


class GomangaApiClient:
    """Client for GOMANGA-API. Search, chapter list, and download via imageUrls."""

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

    def search(self, title: str, limit: int = 20) -> List[dict]:
        """
        Search manga by title. Uses GET /api/search/{keyword}.
        Returns list of dicts with keys: id (slug), title.
        """
        if not title or not title.strip():
            return []
        try:
            # API expects value in path; URL-encode the search term
            keyword = quote(title.strip(), safe="")
            url = f"{BASE_URL}/api/search/{keyword}"
            r = self._get(url)
            r.raise_for_status()
            data = r.json()
            manga_list = data.get("manga") or data.get("mangaList") or []
            err_msg = data.get("error")
            if not manga_list and err_msg:
                raise ValueError(
                    "GOMANGA upstream returned an error (e.g. 403). Try MangaDex or Comick."
                )
            out = []
            for item in manga_list[:limit]:
                mid = item.get("id") or item.get("slug")
                if mid is None:
                    continue
                out.append({
                    "id": mid,
                    "title": item.get("title") or "?",
                    "description": "",
                })
            return out
        except requests.RequestException as e:
            LOGGER.warning(f"GOMANGA search failed: {e}")
            return []
        except ValueError:
            raise
        except Exception as e:
            LOGGER.warning(f"GOMANGA search failed: {e}")
            return []

    def get_feed(
        self,
        manga_id: str,
        translated_language: str = "en",
        limit: int = 500,
        order: str = "asc",
    ) -> List[dict]:
        """
        Get chapter list for a manga. GET /api/manga/{mangaId}.
        Returns list of dicts with keys: id (manga_slug|chapter_num), chapter, display.
        translated_language and order are ignored (API does not support them).
        """
        if not manga_id or not str(manga_id).strip():
            return []
        slug = str(manga_id).strip()
        try:
            r = self._get(f"{BASE_URL}/api/manga/{slug}")
            r.raise_for_status()
            data = r.json()
            chapters_raw = data.get("chapters") or []
            out = []
            for ch in chapters_raw:
                ch_id = ch.get("chapterId")
                if ch_id is None:
                    continue
                composite_id = f"{slug}{GOMANGA_ID_SEP}{ch_id}"
                display = f"Ch.{ch_id}"
                out.append({
                    "id": composite_id,
                    "chapter": ch_id,
                    "volume": None,
                    "title": "",
                    "display": display,
                })
            # API returns newest first; we want asc by chapter number
            def sort_key(x):
                try:
                    return (0, float(x.get("chapter") or 0))
                except (TypeError, ValueError):
                    return (1, 0)
            out.sort(key=sort_key)
            if order != "asc":
                out.reverse()
            return out[:limit]
        except Exception as e:
            LOGGER.warning(f"GOMANGA feed failed: {e}")
            return []

    def get_chapter_images(self, manga_slug: str, chapter_num: str) -> Optional[List[str]]:
        """
        Get image URLs for a chapter. GET /api/manga/{mangaId}/{chapter}.
        Returns list of image URLs or None on failure.
        """
        if not manga_slug or not chapter_num:
            return None
        try:
            r = self._get(f"{BASE_URL}/api/manga/{manga_slug}/{chapter_num}")
            r.raise_for_status()
            data = r.json()
            urls = data.get("imageUrls")
            if isinstance(urls, list) and urls:
                return urls
            return None
        except Exception as e:
            LOGGER.warning(f"GOMANGA chapter images failed: {e}")
            return None

    def download_chapter(
        self,
        chapter_id: str,
        save_dir: str,
        data_saver: bool = False,
        on_progress: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Download a chapter's images to save_dir. Saves as 001.ext, 002.ext, ...
        chapter_id must be "manga_slug|chapter_num" (from get_feed).
        data_saver is ignored (API returns single quality). Returns save_dir on success.
        """
        if not chapter_id or GOMANGA_ID_SEP not in chapter_id:
            return None
        parts = chapter_id.split(GOMANGA_ID_SEP, 1)
        manga_slug = parts[0].strip()
        chapter_num = parts[1].strip()
        if not manga_slug or not chapter_num:
            return None
        urls = self.get_chapter_images(manga_slug, chapter_num)
        if not urls:
            return None
        os.makedirs(save_dir, exist_ok=True)
        total = len(urls)
        for i, url in enumerate(urls):
            try:
                r = self._get(url)
                r.raise_for_status()
                ext = Path(url).suffix or ".webp"
                if not ext.startswith("."):
                    ext = "." + ext
                page_name = f"{i + 1:03d}{ext}"
                path = osp.join(save_dir, page_name)
                with open(path, "wb") as f:
                    f.write(r.content)
                if on_progress:
                    on_progress(i + 1, total, page_name)
            except Exception as e:
                LOGGER.warning(f"GOMANGA download page {i + 1} failed: {e}")
                return None
        return save_dir
