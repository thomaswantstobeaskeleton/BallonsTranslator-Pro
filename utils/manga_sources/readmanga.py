"""ReadManga-style static provider.

Best-effort clean-room parser for public ReadManga-like pages. No authentication,
private API, or access-control bypass is implemented.
"""
from __future__ import annotations

from .static_site import StaticHtmlMangaProvider


class ReadMangaClient(StaticHtmlMangaProvider):
    source_id = "readmanga"
    display_name = "ReadManga-style"
    base_url = "https://readmanga.live"
    search_path_template = "/search?q={query}"
    search_item_selector = ".tiles .tile, .manga-list .item, .series, .book-item"
    search_link_selector = "a[href]"
    search_title_selector = ".name, .title, h3, a"
    feed_url_template = "/{manga_id}"
    feed_chapter_selector = ".chapters-link a, .chapter-list a, a[href*='/vol'], a[href*='/chapter']"
    page_image_selector = ".page img, .reader img, img[data-src], img[src]"
    manga_path_markers = ("/manga/", "/read/", "/")
    chapter_path_markers = ("/vol", "chapter", "read")
