"""Registry for manga source providers.

The registry keeps UI source metadata separate from the dialog, so adding a new
provider no longer requires hand-editing the source combo box.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib import import_module
import requests
from typing import Any, Callable, Optional

from utils.logger import logger as LOGGER

from .provider_base import MangaSourceCapabilities, MangaSourceProvider


@dataclass(frozen=True)
class SourceMetadata:
    source_id: str
    display_name: str
    provider_path: str | None = None
    base_url: str = ""
    status: str = "enabled"  # enabled, disabled, experimental, broken
    capabilities: MangaSourceCapabilities = field(default_factory=MangaSourceCapabilities)
    aliases: tuple[str, ...] = ()
    fallback_base_urls: tuple[str, ...] = ()
    category: str = "english aggregator"
    notes: str = ""
    legacy: bool = True
    config_base_url_key: str | None = None

    @property
    def requires_playwright(self) -> bool:
        return self.capabilities.requires_playwright

    @property
    def requires_cookies(self) -> bool:
        return self.capabilities.supports_cookies and self.status == "disabled"

    def label(self) -> str:
        badges: list[str] = []
        if self.status == "experimental":
            badges.append("experimental")
        elif self.status == "broken":
            badges.append("broken")
        elif self.status == "disabled":
            badges.append("disabled")
        if self.capabilities.requires_playwright:
            badges.append("needs browser")
        if self.capabilities.supports_cookies:
            badges.append("cookies")
        return f"{self.display_name} ({', '.join(badges)})" if badges else self.display_name


_PROVIDER_CACHE: dict[str, type[MangaSourceProvider] | None] = {}
_SOURCES: dict[str, SourceMetadata] = {}
_ALIASES: dict[str, str] = {}


def register_source(metadata: SourceMetadata) -> None:
    _SOURCES[metadata.source_id] = metadata
    for alias in metadata.aliases:
        _ALIASES[alias] = metadata.source_id


def _load_provider_class(path: str) -> type[MangaSourceProvider] | None:
    if path in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[path]
    try:
        module_name, class_name = path.rsplit(":", 1)
        module = import_module(module_name)
        cls = getattr(module, class_name)
        _PROVIDER_CACHE[path] = cls
        return cls
    except Exception as exc:  # optional deps or broken provider must not crash app startup
        LOGGER.warning("Failed to import manga source provider %s: %s", path, exc)
        _PROVIDER_CACHE[path] = None
        return None


def list_sources(include_disabled: bool = False, include_experimental: bool = True) -> list[SourceMetadata]:
    out: list[SourceMetadata] = []
    for meta in _SOURCES.values():
        if not include_disabled and meta.status in {"disabled", "broken"}:
            continue
        if not include_experimental and meta.status == "experimental":
            continue
        out.append(meta)
    return out


def get_metadata(source_id: str) -> SourceMetadata:
    resolved = _ALIASES.get(source_id, source_id)
    if resolved not in _SOURCES:
        raise KeyError(f"Unknown manga source: {source_id}")
    return _SOURCES[resolved]


def get_provider(source_id: str, config: Any = None, **kwargs: Any) -> MangaSourceProvider:
    meta = get_metadata(source_id)
    if not meta.provider_path:
        raise KeyError(f"Manga source {source_id} has no provider class")
    cls = _load_provider_class(meta.provider_path)
    if cls is None:
        raise RuntimeError(f"Manga source {source_id} could not be imported")
    base_url = kwargs.pop("base_url", None) or meta.base_url
    if config is not None:
        overrides = getattr(config, "manga_source_base_url_overrides", {}) or {}
        if isinstance(overrides, dict):
            base_url = (overrides.get(meta.source_id) or base_url).strip()
        if meta.config_base_url_key:
            base_url = (getattr(config, meta.config_base_url_key, "") or base_url).strip()
    return cls(base_url=base_url, **kwargs)


def source_options(include_disabled: bool = False) -> list[tuple[str, str]]:
    return [(m.label(), m.source_id) for m in list_sources(include_disabled=include_disabled)]


CAP_SEARCH_DOWNLOAD = MangaSourceCapabilities(supports_search=True, supports_download=True, supports_referer=True)
CAP_GENERIC_URL = MangaSourceCapabilities(supports_search=False, supports_chapter_url=True, supports_download=True, requires_playwright=False, supports_referer=True)
CAP_MANGADEX = MangaSourceCapabilities(supports_search=True, supports_latest=False, supports_download=True, supports_raw_language=True)
CAP_SEARCH_ONLY = MangaSourceCapabilities(supports_search=True, supports_download=False)


# Preserve the exact legacy display names and source IDs first.
for meta in [
    SourceMetadata("mangadex", "MangaDex", "utils.manga_sources.mangadex:MangaDexClient", "https://api.mangadex.org", capabilities=CAP_MANGADEX, category="bridge/API"),
    SourceMetadata("mangadex_raw", "MangaDex (raw / original language)", "utils.manga_sources.mangadex:MangaDexClient", "https://api.mangadex.org", capabilities=CAP_MANGADEX, category="Japanese raw", notes="Uses MangaDex originalLanguage and translatedLanguage filters."),
    SourceMetadata("mangadex_url", "MangaDex (by chapter URL)", "utils.manga_sources.mangadex:MangaDexClient", "https://api.mangadex.org", capabilities=MangaSourceCapabilities(supports_search=False, supports_chapter_url=True, supports_download=True), category="generic URL"),
    SourceMetadata("comick", "Comick", "utils.manga_sources.comick_source:ComickSourceClient", "https://comick-source-api.notaspider.dev", capabilities=CAP_SEARCH_ONLY, category="bridge/API", notes="Search and chapter list only; API does not expose page image URLs."),
    SourceMetadata("gomanga", "GOMANGA", "utils.manga_sources.gomanga_api:GomangaApiClient", "https://gomanga-api.vercel.app", capabilities=CAP_SEARCH_DOWNLOAD, category="bridge/API"),
    SourceMetadata("manhwa_reader", "Manhwa Reader", "utils.manga_sources.manhwa_reader:ManhwaReaderClient", "https://manhwa-reader-api.vercel.app", capabilities=CAP_SEARCH_DOWNLOAD, category="Korean raw"),
    SourceMetadata("mangaforfree", "MangaForFree", "utils.manga_sources.mangaforfree:MangaForFreeClient", "https://mangaforfree.com", capabilities=MangaSourceCapabilities(requires_playwright=False), fallback_base_urls=("https://mangaforfree.com",)),
    SourceMetadata("toongod", "ToonGod", "utils.manga_sources.toongod:ToonGodClient", "https://toongod.org", capabilities=MangaSourceCapabilities(requires_playwright=False), category="Korean raw"),
    SourceMetadata("mangakakalot", "Mangakakalot", "utils.manga_sources.mangakakalot:MangakakalotClient", "https://www.mangakakalot.gg", capabilities=MangaSourceCapabilities(requires_playwright=False), aliases=("mangakakalot_gg",)),
    SourceMetadata("naruraw", "NaruRaw (Japanese raw)", "utils.manga_sources.naruraw:NaruRawClient", "https://naruraw.net", capabilities=MangaSourceCapabilities(supports_raw_language=True), category="Japanese raw"),
    SourceMetadata("manhwaraw", "ManhwaRaw (Korean raw)", "utils.manga_sources.manhwaraw:ManhwaRawClient", "https://manhwaraw.club", capabilities=MangaSourceCapabilities(supports_raw_language=True), category="Korean raw"),
    SourceMetadata("onekkk", "1kkk (Chinese manhua)", "utils.manga_sources.onekkk:OneKkkClient", "https://www.1kkk.com", capabilities=MangaSourceCapabilities(supports_raw_language=True), category="Chinese manhua"),
    SourceMetadata("generic_chapter_url", "Generic (chapter URL)", "utils.manga_sources.generic_chapter_url:GenericChapterUrlClient", "", capabilities=CAP_GENERIC_URL, category="generic URL"),
    SourceMetadata("raws_manhwa_manhua_url", "Raws / Manhwa / Manhua (chapter URL)", "utils.manga_sources.generic_chapter_url:GenericChapterUrlClient", "", capabilities=CAP_GENERIC_URL, aliases=("raw_url",), category="generic URL"),
    SourceMetadata("local_folder", "Local folder", None, "", status="enabled", capabilities=MangaSourceCapabilities(supports_search=False, supports_download=False), category="local folder"),
    # First clean-room provider batch; static/public page parsers, no bypass logic.
    SourceMetadata("mangasee", "MangaSee / MangaSee123", "utils.manga_sources.mangasee:MangaSeeClient", "https://mangasee123.com", capabilities=CAP_SEARCH_DOWNLOAD, aliases=("mangasee123",), fallback_base_urls=("https://mangasee123.com", "https://mangaseeonline.us"), notes="Static reader parser; no login/cookie bypass."),
    SourceMetadata("readmanga", "ReadManga-style", "utils.manga_sources.readmanga:ReadMangaClient", "https://readmanga.live", capabilities=CAP_SEARCH_DOWNLOAD, fallback_base_urls=("https://readmanga.live",), category="english aggregator", notes="Best-effort static HTML parser; use only public chapters."),
    SourceMetadata("manganato", "Manganato / Manganelo", "utils.manga_sources.manganato:ManganatoClient", "https://manganato.com", capabilities=CAP_SEARCH_DOWNLOAD, aliases=("manganelo", "chapmanganato"), fallback_base_urls=("https://manganato.com", "https://chapmanganato.to"), notes="Compatibility provider for public Manganato/Manganelo pages."),
    SourceMetadata("rawkuma", "RawKuma (Japanese raw)", "utils.manga_sources.rawkuma:RawKumaClient", "https://rawkuma.com", status="experimental", capabilities=MangaSourceCapabilities(supports_search=True, supports_download=True, supports_raw_language=True), category="Japanese raw", notes="Experimental public parser for RawKuma-style pages."),
    SourceMetadata("manhuagui", "Manhuagui (Chinese manhua)", "utils.manga_sources.manhuagui:ManhuaguiClient", "https://www.manhuagui.com", status="experimental", capabilities=MangaSourceCapabilities(supports_search=True, supports_download=True, supports_raw_language=True), category="Chinese manhua", notes="Experimental public parser for Manhuagui-style pages."),
    SourceMetadata("suwayomi_bridge", "Suwayomi bridge", None, "http://127.0.0.1:4567", status="disabled", capabilities=MangaSourceCapabilities(supports_search=True, supports_download=True, supports_cookies=False), category="bridge/API", notes="Optional future bridge; disabled until configured."),
]:
    register_source(meta)


def check_source_health(source_id: str, config: Any = None, timeout: int = 8) -> tuple[bool, str]:
    """Best-effort source health check using public base URL reachability."""
    try:
        meta = get_metadata(source_id)
    except Exception as exc:
        return False, str(exc)
    url = meta.base_url
    if config is not None:
        overrides = getattr(config, "manga_source_base_url_overrides", {}) or {}
        if isinstance(overrides, dict):
            url = (overrides.get(meta.source_id) or url).strip()
    if not url:
        return True, "No base URL required for this source."
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "BallonsTranslator/1.0"})
        if 200 <= r.status_code < 400:
            return True, f"HTTP {r.status_code} OK: {url}"
        return False, f"HTTP {r.status_code}: {url}"
    except Exception as exc:
        return False, f"Request failed: {exc}"
