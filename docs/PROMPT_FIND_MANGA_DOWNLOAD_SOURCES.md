# Prompt: Find Manga Sources With Download Support for BallonsTranslator

Use the prompt below with ChatGPT (or another LLM) to get a list of manga/comic APIs or sources that support **downloading chapter images** (not just search + chapter list). Copy the entire block into the chat.

---

## Prompt (copy from here)

I'm adding manga/comic sources to an open-source desktop app (BallonsTranslator) that lets users search for manga, list chapters, and **download chapter page images** to a folder (e.g. 001.png, 002.png) for local translation. I need **sources that support full download** — i.e. the API or service must provide **direct image URLs** (or a documented way to obtain them) for each page in a chapter, not just links to a reader webpage.

**What we already have**
- **MangaDex**: Full support. Public API: search manga, get chapter feed by language, get chapter by ID, and **at-home API** returns base URL + list of image filenames per chapter so we can download pages. This is the ideal pattern.
- **Comick Source API** (comick-source-api.notaspider.dev): Search + chapter list only; it returns chapter *reader URLs*, not image URLs, so we cannot implement download for it.

**What I need you to find**
1. **Public or well-documented APIs** (REST/GraphQL) that provide:
   - **Search** (by title or keyword) → list of manga with some ID or slug.
   - **Chapter list** for a manga (by ID/slug) → list of chapters with chapter ID or stable identifier.
   - **Chapter images** for a given chapter: either
     - A single endpoint that returns an array of image URLs (or URLs to image resources), or
     - Documented steps/endpoints to get page image URLs from a chapter ID (e.g. “get chapter → get page list → each page has image URL”).
2. **Preference**: Free, no API key required (or optional key). Prefer official or widely used community APIs.
3. **Legal / ToS**: Prefer sources that are clearly legal (e.g. official publishers, licensed aggregators, or services that allow programmatic access in their terms). Note if a source is unofficial or may have legal gray areas.
4. **Stability**: Note if the API is official, maintained, or used by major apps (e.g. Tachiyomi extensions, Hakuneko, etc.).

**Output format**
For each source you recommend, please provide:
- **Name** of the source / API.
- **Base URL** (e.g. `https://api.example.com`).
- **Documentation link** (official docs, GitHub README, or post that describes the API).
- **Endpoints** (or flow):
  - Search: method + path + main parameters.
  - Chapter list: method + path + how to identify a manga (id/slug/url).
  - Chapter images: method + path + how to get a list of image URLs for a chapter (and any auth or headers needed).
- **Auth**: None / API key in header / cookie / etc.
- **Notes**: Any limitation (rate limits, language, region), and whether it’s official vs unofficial.

**What to exclude**
- Sources that only provide “search + chapter list” or “chapter reader URL” without a way to get **per-page image URLs** (we already have that with Comick Source API).
- Scraping-only approaches with no documented API, unless there is a stable, well-documented scraper/API wrapper (e.g. a single maintained library or public API that does the scraping and exposes endpoints).

**Bonus**
- If you know of **aggregator APIs** (like Comick Source API) that *do* expose chapter image URLs or a “get chapter pages” endpoint, list those too with the exact endpoint and response shape.

Thank you. Please list as many qualifying sources as you can find, with the structure above so I can implement them in Python.

---

## After you get the list

- Add new clients under `utils/manga_sources/` (e.g. `utils/manga_sources/new_source.py`) following the pattern in `mangadex.py`: `search()`, `get_feed()`, `get_chapter_by_id()` (if URL mode), `get_chapter_urls()` or equivalent, and `download_chapter()` that saves pages as `001.ext`, `002.ext`, ...
- Register the source in `utils/manga_sources/__init__.py` and in `ui/manga_source_dialog.py` (source combo + worker branch on `source_id`).
