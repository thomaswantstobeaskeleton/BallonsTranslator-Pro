# Manga Source Provider API

BallonsTranslator-Pro manga/raw sources are registered in `utils/manga_sources/registry.py` and should implement the typed contract in `utils/manga_sources/provider_base.py`.

## Legal and operational rules

Providers must:

- use public pages or documented/public APIs only;
- avoid DRM, paywall, login, private API, CAPTCHA, Cloudflare, or anti-bot bypass;
- use conservative `request_delay` throttling;
- never log cookies, tokens, or secrets;
- keep downloads in the existing folder/page naming shape (`001.ext`, `002.ext`, and `manifest.json` when using the generic downloader);
- clearly mark sources as `experimental`, `disabled`, or `broken` when reliability/legal requirements are uncertain.

Playwright is allowed only as the existing explicit, user-controlled browser mode. Cookies may be supported only when the user intentionally supplies them and they must not be logged.

## Provider types

`provider_base.py` defines:

- `MangaSourceCapabilities`
  - `supports_search`
  - `supports_latest`
  - `supports_chapter_url`
  - `supports_download`
  - `supports_raw_language`
  - `requires_playwright`
  - `supports_cookies`
  - `supports_referer`
- `MangaSearchResult`
- `MangaChapter`
- `MangaPage`
- `MangaSourceProvider`

New providers may return legacy dictionaries from `search()` and `get_feed()` so the current UI remains compatible. Use `MangaSearchResult.as_legacy_dict()` and `MangaChapter.as_legacy_dict()` to avoid drift.

## Minimal provider skeleton

```python
from utils.manga_sources.provider_base import MangaSourceCapabilities, MangaSourceProvider

class ExampleClient(MangaSourceProvider):
    source_id = "example"
    display_name = "Example"
    base_url = "https://example.org"
    capabilities = MangaSourceCapabilities(supports_search=True, supports_download=True)

    def search(self, query: str, limit: int = 20, **opts):
        ...

    def get_feed(self, manga_id: str, limit: int = 500, order: str = "asc", **opts):
        ...

    def get_pages(self, chapter_id: str, **opts):
        ...
```

For normal public HTML sites, prefer subclassing `StaticHtmlMangaProvider` from `utils/manga_sources/static_site.py` and configuring selectors rather than rewriting request, URL, and parser boilerplate.

## Registry metadata

Register a source with `SourceMetadata`:

```python
SourceMetadata(
    "example",
    "Example",
    "utils.manga_sources.example:ExampleClient",
    "https://example.org",
    status="enabled",
    capabilities=MangaSourceCapabilities(supports_search=True, supports_download=True),
    aliases=("example_alias",),
    fallback_base_urls=("https://mirror.example.org",),
    category="English aggregator",
    notes="Public static HTML parser; no login content.",
)
```

Status values:

- `enabled`: shown by default and included in stable smoke tests.
- `experimental`: visible with an experimental badge; future tests should require `--include-experimental`.
- `disabled`: hidden by default; used for optional bridges or sources requiring configuration.
- `broken`: hidden by default until repaired.

The registry resolves aliases, supports fallback base URL metadata, reads `pcfg.manga_source_base_url_overrides`, and catches import-time provider failures so optional dependencies do not crash the app. Experimental sources are controlled via `pcfg.manga_source_show_experimental`. The dialog also persists category filtering in `pcfg.manga_source_filter`.

## UI integration

The source combo in `ui/manga_source_dialog.py` is built from `source_options()` rather than a hand-maintained list. Add source-specific UI branching only when the source needs a genuinely different workflow. Static public HTML providers should use the existing search → feed → generic URL download flow.

## Tests

Required tests for new providers:

1. A mocked parser test with saved HTML fixtures under `tests/fixtures/manga_sources/`.
2. Registry metadata/alias coverage when adding new source IDs.
3. `python -m py_compile` for edited Python files.
4. `pytest -q tests/test_manga_provider_base.py` or a provider-specific test module.
5. Optional network smoke test: `python scripts/test_manga_sources.py --stable-registry-only`.

Network smoke tests are not a substitute for fixture tests because upstream sites can change or block requests.

## External audit workflow

Use `scripts/audit_external_manga_sources.py` to inspect local copies of reference downloader ecosystems. It reads names, domains, languages/ecosystems, path hints, and licenses from local files only and writes `docs/EXTERNAL_SOURCE_AUDIT.md`.

```bash
python scripts/audit_external_manga_sources.py
python scripts/audit_external_manga_sources.py external/hakuneko external/gallery-dl
```
