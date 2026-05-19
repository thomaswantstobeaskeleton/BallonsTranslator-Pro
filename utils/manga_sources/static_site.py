"""Reusable static HTML provider helpers for low-risk manga sources."""
from __future__ import annotations

import json
import re
import time
from typing import Any, Iterable, Optional
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .generic_chapter_url import GenericChapterUrlClient
from .provider_base import MangaChapter, MangaPage, MangaSearchResult, MangaSourceCapabilities, MangaSourceProvider, legacy_chapters, legacy_search_results
from .url_utils import infer_id, normalize_base_url

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def absolute_url(base: str, href: str | None) -> str:
    if not href:
        return ""
    href = href.strip()
    if href.startswith("//"):
        return "https:" + href
    return urljoin(base, href)


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def dedupe_by_url(items: Iterable[MangaSearchResult]) -> list[MangaSearchResult]:
    seen: set[str] = set()
    out: list[MangaSearchResult] = []
    for item in items:
        key = item.url or item.id
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


class StaticHtmlMangaProvider(MangaSourceProvider):
    """Base class for public/static manga sites.

    Subclasses configure URL builders and CSS selectors. The implementation uses
    normal requests with conservative throttling, and can fall back to the shared
    generic chapter downloader for image extraction.
    """

    capabilities = MangaSourceCapabilities(supports_search=True, supports_download=True, supports_referer=True)
    search_path_template = "/search/story/{query}"
    search_item_selector = ""
    search_link_selector = "a[href]"
    search_title_selector = ""
    feed_chapter_selector = "a[href]"
    feed_url_template = "/{manga_id}"
    page_image_selector = "img[src], img[data-src], img[data-original], img[data-lazy-src]"
    manga_path_markers: tuple[str, ...] = ()
    chapter_path_markers: tuple[str, ...] = ("chapter",)

    def __init__(self, base_url: Optional[str] = None, timeout: int = 25, request_delay: float = 0.5):
        super().__init__(base_url=normalize_base_url(base_url or self.base_url), timeout=timeout, request_delay=request_delay)
        self.session.headers["User-Agent"] = USER_AGENT
        self._generic = GenericChapterUrlClient(timeout=timeout, request_delay=request_delay)

    def _throttle(self) -> None:
        if self.request_delay > 0:
            time.sleep(self.request_delay)

    def _get(self, url: str, referer: str | None = None) -> requests.Response:
        self._throttle()
        headers = {"Referer": referer} if referer else None
        return self.session.get(url, timeout=self.timeout, headers=headers)

    def _search_url(self, query: str) -> str:
        return urljoin(self.base_url + "/", self.search_path_template.format(query=quote_plus(query.strip())).lstrip("/"))

    def search(self, query: str, limit: int = 20, **opts: Any) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        response = self._get(self._search_url(query))
        response.raise_for_status()
        return legacy_search_results(self.parse_search(response.text, limit=limit))

    def parse_search(self, html: str, limit: int = 20) -> list[MangaSearchResult]:
        soup = BeautifulSoup(html, "html.parser")
        items = soup.select(self.search_item_selector) if self.search_item_selector else []
        if not items:
            selector = ", ".join(f'a[href*="{marker}"]' for marker in self.manga_path_markers) or "a[href]"
            items = soup.select(selector)
        results: list[MangaSearchResult] = []
        for item in items:
            link = item.select_one(self.search_link_selector) if self.search_link_selector else item
            if link is None and getattr(item, "name", "") == "a":
                link = item
            if not link or not getattr(link, "get", None):
                continue
            href = link.get("href")
            url = absolute_url(self.base_url, href)
            if not self._looks_like_manga_url(url):
                continue
            title_node = item.select_one(self.search_title_selector) if self.search_title_selector else None
            title = clean_text(title_node.get_text(" ") if title_node else link.get("title") or link.get_text(" "))
            manga_id = self.manga_id_from_url(url)
            if not manga_id:
                continue
            results.append(MangaSearchResult(id=manga_id, title=title or manga_id, url=url))
        return dedupe_by_url(results)[:limit]

    def _looks_like_manga_url(self, url: str) -> bool:
        if not url:
            return False
        path = urlparse(url).path.lower()
        if any(skip in path for skip in ("/genre", "/author", "/tag", "/page/", "/privacy")):
            return False
        if self.manga_path_markers:
            return any(marker.lower() in path for marker in self.manga_path_markers)
        return True

    def manga_id_from_url(self, url: str) -> str:
        return infer_id(url)

    def manga_url(self, manga_id: str) -> str:
        if manga_id.startswith("http://") or manga_id.startswith("https://"):
            return manga_id
        return urljoin(self.base_url + "/", self.feed_url_template.format(manga_id=manga_id).lstrip("/"))

    def get_feed(self, manga_id: str, limit: int = 500, order: str = "asc", **opts: Any) -> list[dict[str, Any]]:
        response = self._get(self.manga_url(manga_id))
        response.raise_for_status()
        return legacy_chapters(self.parse_feed(response.text, manga_id=manga_id, limit=limit, order=order))

    def parse_feed(self, html: str, manga_id: str = "", limit: int = 500, order: str = "asc") -> list[MangaChapter]:
        soup = BeautifulSoup(html, "html.parser")
        chapters: list[MangaChapter] = []
        seen: set[str] = set()
        for idx, link in enumerate(soup.select(self.feed_chapter_selector), start=1):
            href = link.get("href")
            url = absolute_url(self.base_url, href)
            if not self._looks_like_chapter_url(url) or url in seen:
                continue
            seen.add(url)
            display = clean_text(link.get("title") or link.get_text(" ")) or f"Chapter {len(chapters) + 1}"
            chapters.append(MangaChapter(id=url, display=display, url=url, index=len(chapters) + 1))
        if order == "desc":
            chapters.reverse()
        return chapters[:limit]

    def _looks_like_chapter_url(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        return bool(url) and any(marker.lower() in path for marker in self.chapter_path_markers)

    def get_pages(self, chapter_id: str, **opts: Any) -> list[MangaPage]:
        response = self._get(chapter_id)
        response.raise_for_status()
        pages = self.parse_pages(response.text, chapter_url=chapter_id)
        if pages:
            return pages
        return [MangaPage(url=u, referer=chapter_id) for u in self._generic.get_chapter_images(chapter_id, use_playwright=False)]

    def parse_pages(self, html: str, chapter_url: str) -> list[MangaPage]:
        soup = BeautifulSoup(html, "html.parser")
        out: list[MangaPage] = []
        seen: set[str] = set()
        for img in soup.select(self.page_image_selector):
            src = img.get("data-src") or img.get("data-original") or img.get("data-lazy-src") or img.get("src")
            url = absolute_url(chapter_url, src)
            if not url or url in seen:
                continue
            if not re.search(r"\.(?:jpe?g|png|webp|gif)(?:[?#].*)?$", url, re.I):
                continue
            seen.add(url)
            out.append(MangaPage(url=url, referer=chapter_url))
        return out

    def download_chapter(self, chapter_id: str, save_dir: str, on_progress=None, **opts: Any):
        # Keep manifest/page naming behavior aligned with existing generic URL downloads.
        self._generic.request_delay = self.request_delay
        return self._generic.download_chapter(chapter_id, save_dir, on_progress=on_progress, use_playwright=False)
