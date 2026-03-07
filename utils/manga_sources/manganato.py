"""
MangaNato site client (manganato.com / chapmanganato family).
Search and chapter list via HTTP; chapter images use GenericChapterUrlClient (chapter URL).
Optional Playwright for search/feed when site blocks or requires JS.
Ref: mangal provider (div.search-story-item, li.a-h), FMHY manga sources.
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
# MangaNato/Manganelo search results
SEARCH_ITEM = "div.search-story-item"
SEARCH_TITLE_LINK = "a.item-title"
# Chapter list on manga page
CHAPTER_ITEM = "li.a-h a, .chapter-list a[href*='chapter']"
CHAPTER_ITEM_FALLBACK = "a[href*='/chapter-']"


def _absolute_url(base: str, href: str) -> str:
    if not href or not href.strip():
        return ""
    h = href.strip()
    if h.startswith("//"):
        return "https:" + h
    return urljoin(base, h)


class MangaNatoClient:
    """Sync client for MangaNato-style sites (search/story, manga/slug, chapter links)."""

    def __init__(
        self,
        base_url: str = "https://manganato.com",
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
        """Search by keyword. Returns list of {id, title, url}. use_playwright for JS-heavy/blocked sites."""
        if not keyword or not keyword.strip():
            return []
        keyword = keyword.strip()
        # MangaNato search: spaces -> underscores, lowercase
        query = keyword.replace(" ", "_").lower().strip()
        query = quote(query, safe="")
        search_url = urljoin(self.base_url, f"/search/story/{query}")
        html = None
        if use_playwright:
            html = fetch_html_playwright(
                search_url,
                wait_selector="div.search-story-item, a.item-title",
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

        for item in soup.select(SEARCH_ITEM):
            link_el = item.select_one(SEARCH_TITLE_LINK)
            if not link_el:
                continue
            href = link_el.get("href")
            if not href:
                continue
            full_url = _absolute_url(base, str(href))
            # Skip non-manga links
            if "/manga/" not in full_url or "/chapter-" in full_url:
                continue
            path = urlparse(full_url).path.strip("/")
            parts = path.split("/")
            if len(parts) < 2 or parts[0] != "manga":
                continue
            manga_id = parts[1] if len(parts) > 1 else infer_id(full_url)
            title = (link_el.get_text() or "").strip()
            if not title or len(title) > 300:
                title = manga_id
            results.append({"id": manga_id, "title": title, "url": full_url})

        if not results:
            for a in soup.select('a[href*="/manga/"]'):
                href = a.get("href")
                if not href or "/chapter-" in href:
                    continue
                full_url = _absolute_url(base, str(href))
                path = urlparse(full_url).path.strip("/")
                parts = path.split("/")
                if len(parts) < 2 or parts[0] != "manga":
                    continue
                manga_id = parts[1]
                title = (a.get_text() or "").strip()
                if 2 <= len(title) <= 300:
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
        """Get chapter list for a manga. Chapters have id=full chapter URL for generic download."""
        manga_url = urljoin(self.base_url, f"/manga/{manga_id}")
        if not manga_url.endswith("/"):
            manga_url += "/"
        html = None
        if use_playwright:
            html = fetch_html_playwright(
                manga_url,
                wait_selector="li.a-h a, .chapter-list a, a[href*='chapter']",
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
        items = soup.select(CHAPTER_ITEM)
        if not items:
            items = soup.select(CHAPTER_ITEM_FALLBACK)
        for index, item in enumerate(items, start=1):
            link = item.get("href")
            if not link:
                continue
            full_url = _absolute_url(self.base_url, str(link))
            if full_url in seen_urls:
                continue
            if "/chapter-" not in full_url and "/chapter_" not in full_url:
                continue
            seen_urls.add(full_url)
            title = (item.get_text() or "").strip()
            if title.startswith("Vol."):
                title = title.split(" ", 1)[-1] if " " in title else title
            ch_id = infer_id(full_url)
            chapters.append({
                "id": full_url,
                "display": title or ch_id or f"Ch.{index}",
                "index": len(chapters) + 1,
            })
        if order == "desc":
            chapters.reverse()
        return chapters[:limit]
