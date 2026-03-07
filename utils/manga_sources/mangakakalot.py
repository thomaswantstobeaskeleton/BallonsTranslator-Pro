"""
Mangakakalot.gg site client (www.mangakakalot.gg).
Search and chapter list via HTTP or Playwright; chapter images use GenericChapterUrlClient.
URLs: manga /manga/{slug}, chapter /manga/{slug}/chapter-{num}.
Ref: Kotatsu parsers #1522/#1524, FMHY manga sources.
"""
from __future__ import annotations

import re
import time
from typing import List, Optional
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .generic_chapter_url import fetch_html_playwright
from .url_utils import infer_id, normalize_base_url

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# Manga page: /manga/{slug}, Chapter: /manga/{slug}/chapter-{num}
MANGA_PATH = re.compile(r"^/manga/([^/]+)/?$")
CHAPTER_PATH = re.compile(r"^/manga/[^/]+/chapter-[^/]+/?$")


def _absolute_url(base: str, href: str) -> str:
    if not href or not href.strip():
        return ""
    h = href.strip()
    if h.startswith("//"):
        return "https:" + h
    return urljoin(base, h)


class MangakakalotClient:
    """Sync client for Mangakakalot.gg. Search, feed, download via generic chapter URL."""

    def __init__(
        self,
        base_url: str = "https://www.mangakakalot.gg",
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
        """Search by keyword. Tries /search/story/{query} then /search?q=."""
        if not keyword or not keyword.strip():
            return []
        keyword = keyword.strip()
        query_slug = keyword.replace(" ", "_").lower()
        query_enc = quote(keyword)
        urls_to_try = [
            urljoin(self.base_url, f"/search/story/{quote(query_slug)}"),
            urljoin(self.base_url, f"/search?q={query_enc}"),
            urljoin(self.base_url, f"/search/story/{query_enc}"),
        ]
        html = None
        for url in urls_to_try:
            if use_playwright:
                html = fetch_html_playwright(
                    url,
                    wait_selector="a[href*='/manga/']",
                    timeout_ms=20000,
                    headless=headless,
                )
            if not html:
                try:
                    r = self._get(url)
                    r.raise_for_status()
                    html = r.text
                except requests.RequestException:
                    continue
            if html and ("/manga/" in html or "manga" in html.lower()):
                break
        if not html:
            return []
        return self._parse_search_results(html)[:limit]

    def _parse_search_results(self, html: str) -> List[dict]:
        soup = BeautifulSoup(html, "html.parser")
        results: List[dict] = []
        base = self.base_url
        seen: set = set()

        for a in soup.select('a[href*="/manga/"]'):
            href = a.get("href")
            if not href:
                continue
            full_url = _absolute_url(base, str(href))
            path = urlparse(full_url).path.strip("/")
            if "/chapter-" in path or "/genre/" in path or "/manga-list" in path or "/manga_list" in path:
                continue
            m = re.match(r"^manga/([^/]+)/?$", path)
            if not m:
                continue
            slug = m.group(1)
            if slug in seen:
                continue
            seen.add(slug)
            title = (a.get_text() or "").strip()
            if not title or len(title) > 300:
                title = slug.replace("-", " ")
            results.append({"id": slug, "title": title, "url": full_url})

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
        """Get chapter list. manga_id = slug. Chapter id = full URL."""
        slug = (manga_id or "").strip().strip("/")
        if not slug:
            return []
        manga_url = urljoin(self.base_url, f"/manga/{slug}")
        html = None
        if use_playwright:
            html = fetch_html_playwright(
                manga_url,
                wait_selector="a[href*='/chapter-']",
                timeout_ms=20000,
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

        for a in soup.select('a[href*="/manga/"][href*="/chapter-"]'):
            href = a.get("href")
            if not href:
                continue
            full_url = _absolute_url(self.base_url, str(href))
            if full_url in seen_urls:
                continue
            path = urlparse(full_url).path
            if not re.search(r"/manga/[^/]+/chapter-", path):
                continue
            seen_urls.add(full_url)
            title = (a.get_text() or "").strip() or infer_id(full_url) or f"Ch.{len(chapters) + 1}"
            chapters.append({
                "id": full_url,
                "display": title,
                "index": len(chapters) + 1,
            })

        if order == "desc":
            chapters.reverse()
        return chapters[:limit]
