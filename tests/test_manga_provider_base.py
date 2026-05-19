from pathlib import Path

from utils.manga_sources.manganato import ManganatoClient
from utils.manga_sources.mangasee import MangaSeeClient
from utils.manga_sources.provider_base import MangaChapter, MangaPage, MangaSearchResult, MangaSourceCapabilities
from utils.manga_sources.readmanga import ReadMangaClient
from utils.manga_sources.rawkuma import RawKumaClient
from utils.manga_sources.manhuagui import ManhuaguiClient
from utils.manga_sources.registry import get_metadata, get_provider, list_sources, check_source_health

FIXTURES = Path(__file__).parent / "fixtures" / "manga_sources"


def test_provider_dataclasses_legacy_dicts():
    caps = MangaSourceCapabilities(supports_raw_language=True, requires_playwright=True)
    assert caps.supports_raw_language is True
    assert caps.requires_playwright is True
    assert MangaSearchResult("id", "Title", url="https://e.test", extra={"description": "d"}).as_legacy_dict()["description"] == "d"
    assert MangaChapter("c", "Chapter", index=1).as_legacy_dict()["index"] == 1
    assert MangaPage("https://e.test/1.jpg", filename="001.jpg").filename == "001.jpg"


def test_registry_lists_legacy_and_new_sources():
    ids = {item.source_id for item in list_sources()}
    assert "mangadex" in ids
    assert "mangasee" in ids
    assert "readmanga" in ids
    assert "manganato" in ids
    assert "rawkuma" in ids
    assert "manhuagui" in ids
    assert get_metadata("manganelo").source_id == "manganato"
    provider = get_provider("manganato", timeout=1, request_delay=0)
    assert provider.source_id == "manganato"


def test_manganato_fixture_parsers():
    client = ManganatoClient(request_delay=0)
    results = client.parse_search((FIXTURES / "manganato_search.html").read_text())
    assert results[0].title == "Sample Manga"
    chapters = client.parse_feed((FIXTURES / "manganato_feed.html").read_text(), manga_id="manga-abc123")
    assert len(chapters) == 2
    assert chapters[0].url.endswith("chapter-1")
    pages = client.parse_pages((FIXTURES / "pages.html").read_text(), "https://manganato.com/manga-abc123/chapter-1")
    assert [p.url for p in pages] == ["https://cdn.example.org/001.jpg", "https://manganato.com/002.webp"]


def test_mangasee_fixture_parsers():
    client = MangaSeeClient(request_delay=0)
    results = client.parse_search((FIXTURES / "mangasee_search.html").read_text())
    assert results[0].id == "Sample-Series"
    chapters = client.parse_feed((FIXTURES / "mangasee_feed.html").read_text(), manga_id="Sample-Series")
    assert chapters[0].display == "Chapter 1 – Start"
    assert "read-online" in chapters[0].url


def test_readmanga_fixture_parsers():
    client = ReadMangaClient(request_delay=0)
    results = client.parse_search((FIXTURES / "readmanga_search.html").read_text())
    assert results[0].id == "sample-title"
    chapters = client.parse_feed((FIXTURES / "readmanga_feed.html").read_text(), manga_id="sample-title")
    assert [c.display for c in chapters] == ["Chapter 1", "Chapter 2"]


def test_raw_providers_fixture_parsers():
    rawkuma = RawKumaClient(request_delay=0)
    rk_search = rawkuma.parse_search((FIXTURES / "rawkuma_search.html").read_text())
    rk_feed = rawkuma.parse_feed((FIXTURES / "rawkuma_feed.html").read_text(), manga_id="sample-raw")
    assert rk_search and rk_feed

    mhg = ManhuaguiClient(request_delay=0)
    mhg_search = mhg.parse_search((FIXTURES / "manhuagui_search.html").read_text())
    mhg_feed = mhg.parse_feed((FIXTURES / "manhuagui_feed.html").read_text(), manga_id="12345")
    assert mhg_search and mhg_feed


def test_registry_health_check_utility():
    ok, msg = check_source_health("generic_chapter_url")
    assert ok is True
    assert "No base URL" in msg
