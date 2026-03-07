# Manga/comic sources for search and download (e.g. MangaDex API).
# Add more sources here as needed.

from .mangadex import MangaDexClient
from .comick_source import ComickSourceClient
from .gomanga_api import GomangaApiClient
from .manhwa_reader import ManhwaReaderClient
from .generic_chapter_url import GenericChapterUrlClient
from .mangaforfree import MangaForFreeClient
from .toongod import ToonGodClient
from .manganato import MangaNatoClient
from .mangafire import MangaFireClient
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
    "MangaNatoClient",
    "MangaFireClient",
    "NaruRawClient",
    "ManhwaRawClient",
    "OneKkkClient",
]
