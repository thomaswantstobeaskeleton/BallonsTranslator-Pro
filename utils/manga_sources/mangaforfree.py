"""
MangaForFree site client (manhua-translator port).
Search and chapter list via HTTP; chapter images use same pipeline as Generic (chapter URL).
Optional Playwright for search/feed when site blocks or requires JS.
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
SEARCH_ITEM = ".c-tabs-item__content, .page-item-detail"
SEARCH_TITLE = ".post-title"
CHAPTER_ITEM = "li.wp-manga-chapter a, .listing-chapters_wrap a, a[href*='/chapter-']"
CHAPTER_TITLE = ".chapter-title"


def _absolute_url(base: str, href: str) -> str:
    if not href or not href.strip():
        return ""
    h = href.strip()
    if h.startswith("//"):
        return "https:" + h
    return urljoin(base, h)


class MangaForFreeClient:
    """Sync client for MangaForFree-style sites (WP-Manga with /manga/ path)."""

    def __init__(
        self,
        base_url: str = "https://mangaforfree.com",
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

    def _post(self, url: str, data: dict, referer: Optional[str] = None) -> requests.Response:
        self._throttle()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if referer:
            headers["Referer"] = referer
        return self.session.post(url, timeout=self.timeout, headers=headers, data=data)

    def search(self, keyword: str, limit: int = 25, use_playwright: bool = False, headless: bool = True) -> List[dict]:
        """Search by keyword. Returns list of {id, title, url}. Set use_playwright=True for JS/Cloudflare sites. headless=True runs browser in background."""
        if not keyword or not keyword.strip():
            return []
        keyword = keyword.strip()
        url = urljoin(self.base_url, f"/?s={quote(keyword)}&post_type=wp-manga")
        html = None
        if use_playwright:
            html = fetch_html_playwright(url, wait_selector=".c-tabs-item__content, .post-title, a[href*='/manga/']", timeout_ms=25000, headless=headless)
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

        for item in soup.select(SEARCH_ITEM):
            # Prefer the title link (.post-title a) so we get the manga page, not "Top Views" or chapter link
            title_el = item.select_one(SEARCH_TITLE)
            link_el = title_el.select_one("a[href]") if title_el else None
            if not link_el:
                link_el = item.select_one("a[href]")
            link = link_el.get("href") if link_el and hasattr(link_el, "get") else None
            if not link:
                continue
            full_url = _absolute_url(base, str(link))
            # Skip nav/order links and non-manga links
            if "/m_orderby=" in full_url or "manga-genre" in full_url or "manga-author" in full_url:
                continue
            path = urlparse(full_url).path.strip("/")
            parts = path.split("/")
            if len(parts) >= 2 and parts[0] == "manga":
                slug = parts[1]
                if slug in ("page", "genre", "category", "tag", ""):
                    continue
            title = title_el.get_text(strip=True) if title_el else (link_el.get_text(strip=True) if link_el else "")
            manga_id = infer_id(full_url)
            results.append({"id": manga_id, "title": title or manga_id, "url": full_url})

        # Fallback: no cards found, try any manga links that look like title entries
        if not results:
            seen_urls: set = set()
            for a in soup.select('a[href*="/manga/"]'):
                href = a.get("href")
                if not href:
                    continue
                full_url = _absolute_url(base, str(href))
                if "manga-genre" in full_url or "manga-author" in full_url or "m_orderby=" in full_url:
                    continue
                path = urlparse(full_url).path.strip("/")
                parts = path.split("/")
                if len(parts) < 2 or parts[0] != "manga":
                    continue
                slug = parts[1]
                if slug in ("page", "genre", "category", "tag", "") or "/chapter-" in full_url:
                    continue
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)
                title = (a.get_text() or "").strip()
                if len(title) < 2 or len(title) > 200:
                    continue
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
        """Get chapter list for a manga. Set use_playwright=True for JS/Cloudflare sites. headless=True runs browser in background."""
        manga_url = urljoin(self.base_url, f"/manga/{manga_id}/")
        html = None
        if use_playwright:
            html = fetch_html_playwright(
                manga_url,
                wait_selector="li.wp-manga-chapter a, .listing-chapters_wrap a, a[href*='/chapter-']",
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
        numeric_id = self._extract_manga_id(html)
        if numeric_id:
            chapter_html = self._fetch_chapters_ajax(numeric_id)
            if chapter_html:
                html = chapter_html
        return self._parse_chapters(html, manga_id, limit=limit, order=order)

    def _extract_manga_id(self, html: str) -> Optional[str]:
        m = re.search(r'manga_id"\s*:\s*"(\d+)"', html)
        if m:
            return m.group(1)
        m = re.search(r'data-id="(\d+)"', html)
        if m:
            return m.group(1)
        return None

    def _fetch_chapters_ajax(self, numeric_manga_id: str) -> Optional[str]:
        ajax_url = urljoin(self.base_url, "/wp-admin/admin-ajax.php")
        data = {"action": "manga_get_chapters", "manga": numeric_manga_id}
        try:
            r = self._post(ajax_url, data, referer=self.base_url + "/")
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            return None

    def _parse_chapters(self, html: str, manga_id: str, limit: int = 500, order: str = "asc") -> List[dict]:
        soup = BeautifulSoup(html, "html.parser")
        chapters: List[dict] = []
        seen_urls: set = set()
        # Primary selectors (WP-Manga theme)
        items = soup.select(CHAPTER_ITEM)
        # Fallback: any link containing /chapter-
        if not items:
            items = soup.select('a[href*="/chapter-"]')
        for index, item in enumerate(items, start=1):
            link = item.get("href")
            if not link:
                continue
            full_url = _absolute_url(self.base_url, str(link))
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            title_el = item.select_one(CHAPTER_TITLE)
            title = title_el.get_text(strip=True) if title_el else item.get_text(strip=True)
            ch_id = infer_id(full_url)
            chapters.append({
                "id": full_url,
                "display": title or ch_id or f"Ch.{index}",
                "index": len(chapters) + 1,
            })
        if order == "desc":
            chapters.reverse()
        return chapters[:limit]
