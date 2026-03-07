"""
1kkk (www.1kkk.com) client — Chinese manhua.
Search and chapter list via HTTP; chapter images use GenericChapterUrlClient.
Optional Playwright when site requires JS.
"""
from __future__ import annotations

import re
import time
from typing import List, Optional
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .generic_chapter_url import fetch_html_playwright
from .url_utils import normalize_base_url

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _absolute_url(base: str, href: str) -> str:
    if not href or not href.strip():
        return ""
    h = href.strip()
    if h.startswith("//"):
        return "https:" + h
    return urljoin(base, h)


class OneKkkClient:
    """Sync client for 1kkk.com (Chinese manhua)."""

    def __init__(
        self,
        base_url: str = "https://www.1kkk.com",
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

    def search(
        self,
        keyword: str,
        limit: int = 25,
        use_playwright: bool = False,
        headless: bool = True,
    ) -> List[dict]:
        """Search by keyword. Returns list of {id, title, url}. manga_id is e.g. manhua33991."""
        if not keyword or not keyword.strip():
            return []
        keyword = keyword.strip()
        url = urljoin(self.base_url, f"/search?title={quote(keyword)}&language=1")
        html = None
        if use_playwright:
            html = fetch_html_playwright(
                url,
                wait_selector="a[href*='/manhua']",
                timeout_ms=25000,
                headless=headless,
            )
        if not html:
            try:
                r = self._get(url)
                r.raise_for_status()
                html = r.text
            except requests.RequestException:
                return []
        return self._parse_search_results(html or "")[:limit]

    def _parse_search_results(self, html: str) -> List[dict]:
        soup = BeautifulSoup(html, "html.parser")
        results: List[dict] = []
        base = self.base_url
        seen: set = set()

        # Links to manga: /manhua36420/, /manhua33991/. Skip search and genre (manhua-xxx).
        for a in soup.select('a[href*="manhua"]'):
            href = a.get("href")
            if not href:
                continue
            full_url = _absolute_url(base, str(href))
            if "search" in full_url or "manhua-" in full_url:
                continue
            path = urlparse(full_url).path
            # Match /manhua12345/ (numeric id only, no suffix)
            m = re.search(r"/manhua(\d+)/?$", path)
            if not m:
                continue
            manga_id = "manhua" + m.group(1)
            if manga_id in seen:
                continue
            seen.add(manga_id)
            title = (a.get_text() or "").strip()
            if len(title) > 100:
                title = title[:97] + "..."
            results.append({"id": manga_id, "title": title or manga_id, "url": full_url})

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
        """Get chapter list. manga_id is e.g. manhua33991. Chapter id = full URL."""
        raw = (manga_id or "").strip().strip("/")
        if not raw:
            return []
        if not raw.startswith("manhua"):
            raw = "manhua" + raw
        manga_url = urljoin(self.base_url, f"/{raw}/")
        html = None
        if use_playwright:
            html = fetch_html_playwright(
                manga_url,
                wait_selector="a[href*='/ch'], a[href*='/vol'], a[href*='/other']",
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
        return self._parse_chapters(html, limit=limit, order=order)

    def _parse_chapters(self, html: str, limit: int = 500, order: str = "asc") -> List[dict]:
        soup = BeautifulSoup(html, "html.parser")
        chapters: List[dict] = []
        seen_urls: set = set()
        # Chapter links: /ch1022-1419280/, /vol2-42466/, /other1261480/
        for a in soup.select('a[href*="/ch"], a[href*="/vol"], a[href*="/other"]'):
            href = a.get("href")
            if not href:
                continue
            full_url = _absolute_url(self.base_url, str(href))
            if full_url in seen_urls:
                continue
            path = urlparse(full_url).path
            if not re.search(r"/(ch\d|vol\d|other\d)", path):
                continue
            seen_urls.add(full_url)
            title = (a.get_text() or "").strip() or path.split("/")[-1] or f"Ch.{len(chapters) + 1}"
            if len(title) > 120:
                title = title[:117] + "..."
            chapters.append({
                "id": full_url,
                "display": title,
                "index": len(chapters) + 1,
            })

        if order == "desc":
            chapters.reverse()
        return chapters[:limit]
