"""Typed provider contracts for manga/comic source integrations.

The legacy downloader clients in this package return dictionaries for UI
compatibility.  New code should implement :class:`MangaSourceProvider` and may
still expose dict-shaped results through the helper ``as_legacy_dict`` methods.
"""
from __future__ import annotations

import os
import os.path as osp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlparse

import requests


@dataclass(frozen=True)
class MangaSourceCapabilities:
    supports_search: bool = True
    supports_latest: bool = False
    supports_chapter_url: bool = False
    supports_download: bool = True
    supports_raw_language: bool = False
    requires_playwright: bool = False
    supports_cookies: bool = False
    supports_referer: bool = True


@dataclass(frozen=True)
class MangaSearchResult:
    id: str
    title: str
    url: Optional[str] = None
    cover_url: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_legacy_dict(self) -> dict[str, Any]:
        out = {"id": self.id, "title": self.title}
        if self.url:
            out["url"] = self.url
        if self.cover_url:
            out["cover_url"] = self.cover_url
        out.update(self.extra)
        return out


@dataclass(frozen=True)
class MangaChapter:
    id: str
    display: str
    url: Optional[str] = None
    index: Optional[int] = None
    volume: Optional[str] = None
    language: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_legacy_dict(self) -> dict[str, Any]:
        out = {"id": self.id, "display": self.display}
        if self.url:
            out["url"] = self.url
        if self.index is not None:
            out["index"] = self.index
        if self.volume is not None:
            out["volume"] = self.volume
        if self.language is not None:
            out["language"] = self.language
        out.update(self.extra)
        return out


@dataclass(frozen=True)
class MangaPage:
    url: str
    filename: Optional[str] = None
    referer: Optional[str] = None
    headers: dict[str, str] = field(default_factory=dict)
    width: Optional[int] = None
    height: Optional[int] = None


class MangaSourceProvider(ABC):
    """Base class for pluggable manga sources.

    Implementations must use public web pages/APIs only, respect rate limits, and
    avoid DRM, paywall, login, private API, or access-control bypasses.
    """

    source_id: str = ""
    display_name: str = ""
    base_url: str = ""
    capabilities: MangaSourceCapabilities = MangaSourceCapabilities()

    def __init__(self, base_url: Optional[str] = None, timeout: int = 25, request_delay: float = 0.5):
        if base_url:
            self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.request_delay = max(0.0, float(request_delay))
        self.session = requests.Session()

    @abstractmethod
    def search(self, query: str, limit: int = 20, **opts: Any) -> list[dict[str, Any]]:
        """Return legacy-compatible search dictionaries."""

    @abstractmethod
    def get_feed(self, manga_id: str, limit: int = 500, order: str = "asc", **opts: Any) -> list[dict[str, Any]]:
        """Return legacy-compatible chapter dictionaries."""

    def get_pages(self, chapter_id: str, **opts: Any) -> list[MangaPage]:
        """Return page URLs for a chapter. Providers may override this.

        Legacy providers that use GenericChapterUrlClient can keep returning
        chapter URLs in ``get_feed`` and use their existing downloader path.
        """
        raise NotImplementedError(f"{self.source_id} does not expose page extraction")

    def download_chapter(
        self,
        chapter_id: str,
        output_dir: str,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        **opts: Any,
    ) -> Optional[str]:
        """Default downloader for providers that implement ``get_pages``."""
        pages = self.get_pages(chapter_id, **opts)
        if not pages:
            return None
        os.makedirs(output_dir, exist_ok=True)
        total = len(pages)
        for idx, page in enumerate(pages, start=1):
            suffix = Path(urlparse(page.url).path).suffix or ".jpg"
            if suffix.lower().lstrip(".") not in {"jpg", "jpeg", "png", "webp", "gif", "bmp"}:
                suffix = ".jpg"
            filename = page.filename or f"{idx:03d}{suffix}"
            headers = dict(page.headers or {})
            if page.referer:
                headers.setdefault("Referer", page.referer)
            response = self.session.get(page.url, timeout=self.timeout, headers=headers or None)
            response.raise_for_status()
            with open(osp.join(output_dir, filename), "wb") as handle:
                handle.write(response.content)
            if on_progress:
                on_progress(idx, total, filename)
        return output_dir


def legacy_search_results(results: Iterable[MangaSearchResult]) -> list[dict[str, Any]]:
    return [item.as_legacy_dict() for item in results]


def legacy_chapters(chapters: Iterable[MangaChapter]) -> list[dict[str, Any]]:
    return [item.as_legacy_dict() for item in chapters]
