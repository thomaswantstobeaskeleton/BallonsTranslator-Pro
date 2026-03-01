"""
Comick Source API client (https://comick-source-api.notaspider.dev).
Search manga and list chapters across multiple sources (MangaPark, Bato, Comix, etc.).
Download is not supported: the API returns chapter reader URLs, not image URLs.
"""
from __future__ import annotations

import json
import time
from typing import Any, List, Optional

import requests

from utils.logger import logger as LOGGER

BASE_URL = "https://comick-source-api.notaspider.dev"
USER_AGENT = "BallonsTranslator/1.0 (https://github.com/dmMaze/BallonsTranslator)"


def _chapter_display(ch: dict) -> str:
    num = ch.get("number")
    title = (ch.get("title") or "").strip()
    if num is not None:
        parts = [f"Ch.{num}"]
        if title:
            parts.append(title)
        return " – ".join(parts)
    return title or ch.get("url", "?")


class ComickSourceClient:
    """Client for Comick Source API. Search + chapter list only; no chapter download."""

    def __init__(self, timeout: int = 30, request_delay: float = 0.3):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.session.headers["Content-Type"] = "application/json"
        self.timeout = timeout
        self.request_delay = max(0.0, float(request_delay))

    def _throttle(self):
        if self.request_delay > 0:
            time.sleep(self.request_delay)

    def _post(self, path: str, json: dict, **kwargs) -> requests.Response:
        self._throttle()
        return self.session.post(f"{BASE_URL}{path}", json=json, timeout=self.timeout, **kwargs)

    def search(self, title: str, limit: int = 20, source: str = "all") -> List[dict]:
        """
        Search manga. Returns list of dicts with keys: id, title, url (use url as id for get_feed).
        API returns NDJSON (one JSON object per source); we merge all results.
        """
        if not title or not title.strip():
            return []
        try:
            r = self._post("/api/search", {"query": title.strip(), "source": source})
            r.raise_for_status()
            results = self._parse_search_response(r)
            out = []
            seen = set()
            for item in results[: limit * 3]:
                url = item.get("url") or item.get("id")
                if not url or url in seen:
                    continue
                seen.add(url)
                out.append({
                    "id": url,
                    "title": item.get("title") or "?",
                    "url": url,
                    "description": "",
                    "source": "Comick",
                })
                if len(out) >= limit:
                    break
            return out
        except Exception as e:
            LOGGER.warning(f"Comick Source search failed: {e}")
            return []

    def _parse_search_response(self, r: requests.Response) -> List[dict]:
        """Parse NDJSON search response: one JSON object per line, merge all 'results' arrays."""
        merged = []
        for line in r.text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                merged.extend(data.get("results") or [])
            except json.JSONDecodeError:
                continue
        return merged

    def _parse_json(self, r: requests.Response) -> dict:
        """Parse single JSON object; if response has extra data, use first line."""
        try:
            return r.json()
        except requests.exceptions.JSONDecodeError as e:
            if "Extra data" in str(e):
                first_line = r.text.strip().split("\n")[0]
                return json.loads(first_line)
            raise

    def get_feed(
        self,
        manga_id: str,
        translated_language: str = "en",
        limit: int = 500,
        order: str = "asc",
        source: Optional[str] = None,
    ) -> List[dict]:
        """
        Get chapter list for a manga. manga_id should be the manga URL from search.
        Returns list of dicts with keys: id, chapter, title, display, url.
        translated_language and order are ignored (API does not support them).
        """
        if not manga_id or not str(manga_id).strip():
            return []
        url = str(manga_id).strip()
        try:
            body = {"url": url}
            if source:
                body["source"] = source
            r = self._post("/api/chapters", body)
            r.raise_for_status()
            data = self._parse_json(r)
            chapters = data.get("chapters") or []
            out = []
            for ch in chapters:
                out.append({
                    "id": ch.get("url") or ch.get("id"),
                    "chapter": ch.get("number"),
                    "volume": None,
                    "title": (ch.get("title") or "").strip(),
                    "display": _chapter_display(ch),
                    "url": ch.get("url"),
                })
            if order == "asc":
                out.sort(key=lambda x: (x.get("chapter") is None, float(x.get("chapter") or 0)))
            else:
                out.sort(key=lambda x: (x.get("chapter") is None, -float(x.get("chapter") or 0)))
            return out[:limit]
        except Exception as e:
            LOGGER.warning(f"Comick Source chapters failed: {e}")
            return []

    def download_chapter(
        self,
        chapter_id: str,
        save_dir: str,
        data_saver: bool = False,
        on_progress: Optional[Any] = None,
    ) -> Optional[str]:
        """Not supported: API does not provide chapter image URLs. Returns None."""
        return None
