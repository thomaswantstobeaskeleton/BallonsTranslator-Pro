"""
NaruRaw (naruraw.net) client — Japanese raw manga.
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


class NaruRawClient:
    """Sync client for NaruRaw (Japanese raw manga)."""

    def __init__(
        self,
        base_url: str = "https://naruraw.net",
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
        """Search by keyword. Returns list of {id, title, url}. manga_id is the /naru/ slug (e.g. xxx-raw)."""
        if not keyword or not keyword.strip():
            return []
        keyword = keyword.strip()
        url = urljoin(self.base_url, f"/filter?keyword={quote(keyword)}")
        html = None
        if use_playwright:
            html = fetch_html_playwright(
                url,
                wait_selector="a[href*='/naru/']",
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

        # Filter page: each manga block has [JA](url) to manga page; title in ### header. Chapter links have /ja/chapter-.
        for a in soup.select('a[href*="/naru/"]'):
            href = a.get("href")
            if not href:
                continue
            full_url = _absolute_url(base, str(href))
            # Skip chapter links (they contain /ja/chapter-)
            if "/ja/chapter-" in full_url or "/genres/" in full_url:
                continue
            # Manga link: /naru/{slug}/ or /naru/{slug}-raw/
            path = urlparse(full_url).path.strip("/")
            if not path.startswith("naru/"):
                continue
            slug = path[5:]  # after "naru/"
            if not slug or slug in seen:
                continue
            seen.add(slug)
            # Prefer title from next sibling ### (h3) or parent section
            title = (a.get_text() or "").strip()
            if title == "JA" or not title:
                parent = a.find_parent(["div", "article", "section"])
                if parent:
                    h3 = parent.select_one("h3")
                    if h3:
                        title = (h3.get_text() or "").strip()
                if not title or title == "JA":
                    title = slug.replace("-raw", "").replace("-", " ")[:80]
            results.append({"id": slug, "title": title or slug, "url": full_url})

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
        """Get chapter list. manga_id is the slug from search (e.g. xxx-raw). Chapter id = full URL."""
        # Ensure we have a valid path: /naru/{manga_id}/
        slug = manga_id.strip().strip("/")
        if not slug:
            return []
        manga_url = urljoin(self.base_url, f"/naru/{slug}/")
        html = None
        if use_playwright:
            html = fetch_html_playwright(
                manga_url,
                wait_selector="a[href*='/ja/chapter-']",
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

        for a in soup.select('a[href*="/ja/chapter-"]'):
            href = a.get("href")
            if not href:
                continue
            full_url = _absolute_url(self.base_url, str(href))
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            title = (a.get_text() or "").strip() or re.sub(r".*chapter-([^/]+).*", r"Ch.\1", href)
            chapters.append({
                "id": full_url,
                "display": title or f"Ch.{len(chapters) + 1}",
                "index": len(chapters) + 1,
            })

        if order == "desc":
            chapters.reverse()
        return chapters[:limit]
