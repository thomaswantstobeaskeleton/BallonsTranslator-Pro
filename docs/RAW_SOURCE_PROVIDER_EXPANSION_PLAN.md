# Raw Source Provider Expansion Plan

## Current source list

Preserved legacy source IDs and UI names:

| Source ID | Display name | Capability summary |
| --- | --- | --- |
| `mangadex` | MangaDex | public MangaDex API search/feed/download |
| `mangadex_raw` | MangaDex (raw / original language) | MangaDex original-language search/feed filters |
| `mangadex_url` | MangaDex (by chapter URL) | chapter UUID/URL loader |
| `comick` | Comick | search and chapter list only; no image URLs |
| `gomanga` | GOMANGA | API-backed search/feed/download |
| `manhwa_reader` | Manhwa Reader | API-backed search/feed/download, hidden when unavailable |
| `mangaforfree` | MangaForFree | static/WP-Manga search/feed, generic URL download |
| `toongod` | ToonGod | static/WP-Manga search/feed, generic URL download |
| `mangakakalot` | Mangakakalot | static search/feed, generic URL download |
| `naruraw` | NaruRaw (Japanese raw) | raw search/feed, generic URL download |
| `manhwaraw` | ManhwaRaw (Korean raw) | raw search/feed, generic URL download |
| `onekkk` | 1kkk (Chinese manhua) | raw search/feed, generic URL download |
| `generic_chapter_url` | Generic (chapter URL) | URL image extraction/download |
| `raws_manhwa_manhua_url` | Raws / Manhwa / Manhua (chapter URL) | URL image extraction/download |
| `local_folder` | Local folder | opens an existing folder as a project |

New registry-backed first-batch providers:

| Source ID | Display name | Status | Notes |
| --- | --- | --- | --- |
| `mangasee` | MangaSee / MangaSee123 | enabled | public static reader parser, no bypass |
| `readmanga` | ReadManga-style | enabled | best-effort public HTML parser |
| `manganato` | Manganato / Manganelo | enabled | public Manganato/Manganelo compatibility parser |
| `suwayomi_bridge` | Suwayomi bridge | disabled | optional future local bridge |
| `rawkuma` | RawKuma (Japanese raw) | experimental | public parser, opt-in via experimental sources |
| `manhuagui` | Manhuagui (Chinese manhua) | experimental | public parser, opt-in via experimental sources |

## Current source interface shape

Legacy clients are synchronous classes with ad-hoc but similar methods:

- `search(title_or_keyword, limit=...) -> list[dict]`
- `get_feed(manga_id, translated_language=..., limit=..., order=...) -> list[dict]`
- `download_chapter(chapter_id, save_dir, ...) -> Optional[str]` for MangaDex/API clients
- chapter-page sources return chapter URLs and use `GenericChapterUrlClient.download_chapter(...)`

The UI keeps source branching in `ui/manga_source_dialog.py`, stores settings in `pcfg.manga_source_*`, and preserves page naming as `001.ext`, `002.ext`, plus `manifest.json` for generic URL downloads.

## Duplicated logic to extract

- Conservative request throttling and shared user-agent setup.
- Absolute URL normalization, slug inference, chapter display formatting.
- Static HTML search-card parsing.
- Static HTML chapter-list parsing.
- Lazy image URL extraction and referer-aware download.
- Source capability/status metadata that was previously embedded in UI conditionals.
- Playwright/headless visibility decisions for sources that may need user-controlled browser rendering.

## Missing provider capabilities

The old shape did not describe whether a source:

- supports search, latest/feed, chapter URL loading, or download;
- supports original/raw language filters;
- requires Playwright, cookies, or referer headers;
- is stable, experimental, disabled, broken, or only a bridge/API;
- has aliases or fallback base URLs;
- should be hidden when optional dependencies fail to import.

`utils/manga_sources/provider_base.py` now defines typed capabilities and result objects, and `utils/manga_sources/registry.py` owns source metadata.

## Proposed source candidates

Low-risk first candidates are public/static HTML or documented public API sources only:

1. MangaSee/MangaSee123-style public pages.
2. ReadManga-style public pages.
3. Manganato/Manganelo compatibility pages and aliases.
4. ComicK resolver improvements only if public image URLs become available from a documented API or visible HTML.
5. MangaDex improvements using public API behavior: rate-limit handling, optional language/group filters, data/data-saver ordering, and stable filenames.
6. Optional Suwayomi bridge that talks to a user-run local server without bundling Suwayomi or Java.

Excluded by policy: DRM/paywall bypass, login-only content, private mobile APIs, hidden tokens, CAPTCHA/Cloudflare bypass, or automated cookies not explicitly supplied by the user.

## License risks per reference repo

| Reference | Pattern to learn | License risk / handling |
| --- | --- | --- |
| keiyoushi/extensions-source | provider classes, capabilities, language/source metadata | Kotlin extensions are typically Apache-style ecosystem code but source-specific files can vary; do not copy code/selectors wholesale without verifying file license. |
| Suwayomi/Suwayomi-Server | local bridge/runtime pattern | Bridge should use only the user's running server API; do not bundle server code. |
| manga-download/hakuneko | connector-style site isolation | HakuNeko has its own licensing and connector code; use behavior/architecture only, no copy-paste. |
| dazedcat19/FMD2 | broad source catalog and Lua parser patterns | GPL/Lua/Pascal obligations are risky; use as candidate inventory only unless compatibility is confirmed. |
| metafates/mangal and mangal-scrapers | Go CLI plus Lua scraper separation | Use plugin/scraper isolation ideas only; do not embed incompatible scraper code. |
| manga-py/manga-py | Python provider inventory | Archived code may be stale and license-specific; use as historical behavior reference only. |
| mikf/gallery-dl | extractor registry, filename/rate-limit practices | GPL-family obligations likely incompatible for direct copying; clean-room architecture ideas only. |
| mansuf/mangadex-downloader | robust public MangaDex handling | Python MangaDex behavior can inspire public API handling; avoid copying implementation unless license obligations are reviewed. |

## Test plan

- Unit-test provider dataclasses, legacy dict conversion, registry aliases, and import-failure safety.
- Unit-test static parser behavior with saved HTML fixtures for each new provider.
- Keep `scripts/test_manga_sources.py` working for legacy stable sources.
- Expand smoke-test discovery so stable registry providers can be tested without enabling disabled sources.
- Add `--include-experimental` for unstable providers (e.g., rawkuma/manhuagui).
- Compile edited Python files and avoid network requirements in unit tests.
- Never log secrets/cookies; fixtures must not contain copyrighted page images.


### Registry parser smoke examples

- `python scripts/test_manga_sources.py --stable-registry-only` (stable registry providers)
- `python scripts/test_manga_sources.py --stable-registry-only --include-experimental` (includes experimental providers with fixtures)
