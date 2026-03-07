"""
MangaFire site client (mangafire.to / mangafire.co).
Search and chapter list via HTTP or Playwright; chapter images use GenericChapterUrlClient.
Manga URL: /manga/{title}.{id}, Chapter: /read/{title}.{id}/{lang}/chapter-{num}.
Ref: MangaFire-API, FMHY manga sources.
"""
from __future__ import annotations

import re
import time
from typing import List, Optional
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .generic_chapter_url import fetch_html_playwright
from .url_utils import infer_id, normalize_base_url, slugify_keyword

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# MangaFire filter/search page and manga page
SEARCH_ITEM = "a[href*='/manga/']"
MANGAFIRE_MANGA_PATH = re.compile(r"^/manga/([^/]+)$")
CHAPTER_LINK = "a[href*='/read/']"


def _absolute_url(base: str, href: str) -> str:
    if not href or not href.strip():
        return ""
    h = href.strip()
    if h.startswith("//"):
        return "https:" + h
    return urljoin(base, h)


class MangaFireClient:
    """Sync client for MangaFire (mangafire.to). Search via /filter, manga /manga/title.id, chapters /read/..."""

    def __init__(
        self,
        base_url: str = "https://mangafire.to",
        timeout: int = 25,
        request_delay: float = 0.5,
    ):
        self.base_url = normalize_base_url(base_url)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.timeout = timeout
        self.request_delay = max(0.0, float(request_delay))

    def _throttle(self) -> None:
        if self.request_delay > 0:
            time.sleep(self.request_delay)

    def _get(self, url: str, referer: Optional[str] = None) -> requests.Response:
        self._throttle()
        headers = {}
        if referer:
            headers["Referer"] = referer
        return self.session.get(url, timeout=self.timeout, headers=headers or None)

    def search(self, keyword: str, limit: int = 25, use_playwright: bool = False, headless: bool = True) -> List[dict]:
        """Search by keyword. MangaFire often requires JS (VRF); use_playwright=True recommended."""
        if not keyword or not keyword.strip():
            return []
        keyword = keyword.strip()
        search_url = urljoin(self.base_url, f"/filter?keyword={quote(keyword)}&page=1&type=")
        html = None
        if use_playwright:
            html = fetch_html_playwright(
                search_url,
                wait_selector="a[href*='/manga/']",
                timeout_ms=25000,
                headless=headless,
            )
        if not html:
            try:
                r = self._get(search_url)
                r.raise_for_status()
                html = r.text
            except requests.RequestException:
                return []
        return self._parse_search_results(html or "")[:limit]

    def _parse_search_results(self, html: str) -> List[dict]:
        soup = BeautifulSoup(html, "html.parser")
        results: List[dict] = []
        base = self.base_url
        seen_urls: set = set()

        for a in soup.select(SEARCH_ITEM):
            href = a.get("href")
            if not href:
                continue
            full_url = _absolute_url(base, str(href))
            path = urlparse(full_url).path.strip("/")
            m = MANGAFIRE_MANGA_PATH.match("/" + path) if path else None
            if not m:
                if "/read/" in full_url or "/filter" in full_url or "/type/" in full_url or "/genre/" in full_url:
                    continue
                if "/manga/" not in full_url:
                    continue
                parts = path.split("/")
                if len(parts) < 2 or parts[0] != "manga":
                    continue
                manga_id = parts[1]
            else:
                manga_id = m.group(1)
            if manga_id in seen_urls or full_url in seen_urls:
                continue
            seen_urls.add(manga_id)
            seen_urls.add(full_url)
            title = (a.get_text() or "").strip()
            if not title or len(title) > 300:
                title = manga_id
            results.append({"id": manga_id, "title": title, "url": full_url})

        return results

    def get_feed(
        self,
        manga_id: str,
        translated_language: Optional[str] = None,
        limit: int = 500,
        order: str = "asc",
        use_playwright: bool = False,
        headless: bool = True,
    ) -> List[dict]:
        """Get chapter list. Chapters have id=full chapter URL for generic download."""
        manga_url = urljoin(self.base_url, f"/manga/{manga_id}")
        html = None
        if use_playwright:
            html = fetch_html_playwright(
                manga_url,
                wait_selector="a[href*='/read/']",
                timeout_ms=25000,
                headless=headless,
            )
        if not html:
            try:
                r = self._get(manga_url)
                r.raise_for_status()
                html = r.text
            except requests.RequestException:
                return []
        if not html:
            return []
        return self._parse_chapters(html, manga_id, limit=limit, order=order)

    def _parse_chapters(self, html: str, manga_id: str, limit: int = 500, order: str = "asc") -> List[dict]:
        soup = BeautifulSoup(html, "html.parser")
        chapters: List[dict] = []
        seen_urls: set = set()

        for a in soup.select(CHAPTER_LINK):
            href = a.get("href")
            if not href:
                continue
            full_url = _absolute_url(self.base_url, str(href))
            if full_url in seen_urls:
                continue
            if "/read/" not in full_url:
                continue
            # Skip "Start Reading" / nav links that don't have chapter number
            if "/chapter-" not in full_url and "/chapter_" not in full_url:
                continue
            seen_urls.add(full_url)
            title = (a.get_text() or "").strip()
            ch_id = infer_id(full_url)
            chapters.append({
                "id": full_url,
                "display": title or ch_id or f"Ch.{len(chapters) + 1}",
                "index": len(chapters) + 1,
            })
        if order == "desc":
            chapters.reverse()
        return chapters[:limit]
