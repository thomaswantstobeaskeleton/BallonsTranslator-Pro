"""MangaSee / MangaSee123 provider.

Uses public HTML/embedded metadata patterns only and falls back to visible reader
images. No anti-bot or access-control bypass is included.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from .provider_base import MangaChapter, MangaPage, MangaSearchResult, legacy_chapters
from .static_site import StaticHtmlMangaProvider, absolute_url, clean_text, dedupe_by_url


class MangaSeeClient(StaticHtmlMangaProvider):
    source_id = "mangasee"
    display_name = "MangaSee / MangaSee123"
    base_url = "https://mangasee123.com"
    search_path_template = "/search/?name={query}"
    search_item_selector = ".SeriesName, .list-group-item, .series-list a, a[href*='/manga/']"
    search_link_selector = "a[href]"
    search_title_selector = ""
    feed_url_template = "/manga/{manga_id}"
    feed_chapter_selector = "a[href*='/read-online/'], a[href*='chapter']"
    page_image_selector = "#readerarea img, .ImageGallery img, img[src], img[data-src]"
    manga_path_markers = ("/manga/",)
    chapter_path_markers = ("/read-online/", "chapter")

    def parse_search(self, html: str, limit: int = 20) -> list[MangaSearchResult]:
        # MangaSee often embeds an IndexName/IndexInfo JSON-ish search index.
        embedded = re.search(r"vm\.IndexName\s*=\s*(\[.*?\]);", html, re.S)
        results: list[MangaSearchResult] = []
        if embedded:
            try:
                for item in json.loads(embedded.group(1)):
                    slug = item.get("i") or item.get("IndexName") or item.get("s")
                    title = item.get("s") or item.get("SeriesName") or slug
                    if slug:
                        url = f"{self.base_url.rstrip('/')}/manga/{slug}"
                        results.append(MangaSearchResult(id=str(slug), title=str(title or slug), url=url))
            except Exception:
                results = []
        if not results:
            results = super().parse_search(html, limit=limit)
        return dedupe_by_url(results)[:limit]

    def parse_feed(self, html: str, manga_id: str = "", limit: int = 500, order: str = "asc") -> list[MangaChapter]:
        chapters: list[MangaChapter] = []
        embedded = re.search(r"vm\.Chapters\s*=\s*(\[.*?\]);", html, re.S)
        if embedded:
            try:
                for idx, item in enumerate(json.loads(embedded.group(1)), start=1):
                    chapter = str(item.get("Chapter") or item.get("chapter") or idx)
                    name = clean_text(item.get("ChapterName") or item.get("name") or "")
                    slug = manga_id or self.manga_id_from_url(self.manga_url(manga_id))
                    url = f"{self.base_url.rstrip('/')}/read-online/{slug}-chapter-{chapter}.html"
                    display = f"Chapter {chapter}" + (f" – {name}" if name else "")
                    chapters.append(MangaChapter(id=url, display=display, url=url, index=idx))
            except Exception:
                chapters = []
        if not chapters:
            chapters = super().parse_feed(html, manga_id=manga_id, limit=limit, order=order)
        if order == "desc":
            chapters.reverse()
        return chapters[:limit]

    def parse_pages(self, html: str, chapter_url: str) -> list[MangaPage]:
        pages = super().parse_pages(html, chapter_url)
        if pages:
            return pages
        # Some mirrors expose page arrays in simple JS variables.
        out: list[MangaPage] = []
        seen: set[str] = set()
        for match in re.finditer(r"https?://[^'\"]+?\.(?:jpg|jpeg|png|webp)(?:\?[^'\"]*)?", html, re.I):
            url = match.group(0)
            if url in seen:
                continue
            seen.add(url)
            out.append(MangaPage(url=url, referer=chapter_url))
        return out
