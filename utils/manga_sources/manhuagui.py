"""Manhuagui-style provider (experimental Chinese manhua source)."""
from __future__ import annotations

from .static_site import StaticHtmlMangaProvider


class ManhuaguiClient(StaticHtmlMangaProvider):
    source_id = "manhuagui"
    display_name = "Manhuagui (Chinese manhua)"
    base_url = "https://www.manhuagui.com"
    search_path_template = "/s/{query}.html"
    search_item_selector = ".book-result li, .result-list li, a[href*='/comic/']"
    search_link_selector = "a[href]"
    search_title_selector = ".book-title, .name, a"
    feed_url_template = "/comic/{manga_id}/"
    feed_chapter_selector = ".chapter-list a, #chapter-list a, a[href*='/comic/'][href*='.html']"
    page_image_selector = "#mangaFile img, .reader-img img, img[data-src], img[src]"
    manga_path_markers = ("/comic/",)
    chapter_path_markers = ("/comic/", ".html")
