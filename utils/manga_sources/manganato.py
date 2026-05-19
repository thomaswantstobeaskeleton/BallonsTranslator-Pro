"""Manganato / Manganelo compatibility provider.

Clean-room parser for public search, chapter list, and reader pages. It does not
perform login, paywall, DRM, Cloudflare, or anti-bot bypass.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from .static_site import StaticHtmlMangaProvider


class ManganatoClient(StaticHtmlMangaProvider):
    source_id = "manganato"
    display_name = "Manganato / Manganelo"
    base_url = "https://manganato.com"
    search_path_template = "/search/story/{query}"
    search_item_selector = ".search-story-item, .panel-search-story .story_item, .story_item"
    search_link_selector = "a.item-img, .item-title a, a[href]"
    search_title_selector = ".item-title, h3 a, a.item-title"
    feed_url_template = "/manga/{manga_id}"
    feed_chapter_selector = ".row-content-chapter li a, .chapter-list .row a, a[href*='chapter']"
    page_image_selector = ".container-chapter-reader img, .vung-doc img, img[src], img[data-src]"
    manga_path_markers = ("/manga-", "/manga/", "/manga_")
    chapter_path_markers = ("chapter",)

    def manga_id_from_url(self, url: str) -> str:
        path = urlparse(url).path.strip("/")
        parts = [p for p in path.split("/") if p]
        return parts[-1] if parts else super().manga_id_from_url(url)

    def manga_url(self, manga_id: str) -> str:
        if manga_id.startswith(("http://", "https://")):
            return manga_id
        if manga_id.startswith("manga-") or manga_id.startswith("manga_"):
            return f"{self.base_url.rstrip('/')}/{manga_id}"
        return f"{self.base_url.rstrip('/')}/manga/{manga_id}"
