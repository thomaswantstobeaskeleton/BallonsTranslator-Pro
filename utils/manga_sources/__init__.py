# Manga/comic sources for search and download (e.g. MangaDex API).
# Add more sources here as needed.

from .mangadex import MangaDexClient
from .comick_source import ComickSourceClient
from .gomanga_api import GomangaApiClient
from .manhwa_reader import ManhwaReaderClient
from .generic_chapter_url import GenericChapterUrlClient
from .mangaforfree import MangaForFreeClient
from .toongod import ToonGodClient
from .mangakakalot import MangakakalotClient
from .naruraw import NaruRawClient
from .manhwaraw import ManhwaRawClient
from .onekkk import OneKkkClient

__all__ = [
    "MangaDexClient",
    "ComickSourceClient",
    "GomangaApiClient",
    "ManhwaReaderClient",
    "GenericChapterUrlClient",
    "MangaForFreeClient",
    "ToonGodClient",
    "MangakakalotClient",
    "NaruRawClient",
    "ManhwaRawClient",
    "OneKkkClient",
    "MangaSourceCapabilities",
    "MangaSearchResult",
    "MangaChapter",
    "MangaPage",
    "MangaSourceProvider",
    "get_provider",
    "list_sources",
    "source_options",
    "check_source_health",
    "MangaSeeClient",
    "ReadMangaClient",
    "ManganatoClient",
    "RawKumaClient",
    "ManhuaguiClient",
]

from .provider_base import (
    MangaChapter,
    MangaPage,
    MangaSearchResult,
    MangaSourceCapabilities,
    MangaSourceProvider,
)
from .registry import get_provider, list_sources, source_options, check_source_health
from .mangasee import MangaSeeClient
from .readmanga import ReadMangaClient
from .manganato import ManganatoClient

from .rawkuma import RawKumaClient
from .manhuagui import ManhuaguiClient
