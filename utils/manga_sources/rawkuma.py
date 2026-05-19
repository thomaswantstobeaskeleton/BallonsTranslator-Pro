"""RawKuma-style provider (experimental).
Public-page parser only; no login/paywall/captcha bypass.
"""
from __future__ import annotations

from .static_site import StaticHtmlMangaProvider


class RawKumaClient(StaticHtmlMangaProvider):
    source_id = "rawkuma"
    display_name = "RawKuma (Japanese raw)"
    base_url = "https://rawkuma.com"
    search_path_template = "/?s={query}&post_type=wp-manga"
    search_item_selector = ".c-tabs-item__content, .page-item-detail, .search-story-item"
    search_link_selector = ".post-title a, a[href]"
    search_title_selector = ".post-title, .item-title, h3"
    feed_url_template = "/manga/{manga_id}/"
    feed_chapter_selector = "li.wp-manga-chapter a, .listing-chapters_wrap a, a[href*='/chapter-']"
    page_image_selector = ".reading-content img, .container-chapter-reader img, img[data-src], img[src]"
    manga_path_markers = ("/manga/",)
    chapter_path_markers = ("chapter",)
