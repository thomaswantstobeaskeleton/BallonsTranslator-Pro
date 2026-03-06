# Manhua-Translator Integration Research

[manhua-translator](https://github.com/aakaka525-design/manhua-translator) is a comic translation stack (OCR, AI translation, inpainting, rendering) with a **built-in web crawler** for downloading manga/manhua chapters. This document summarizes their features and what BallonsTranslator has adopted or can use.

## Their Feature Set (Summary)

| Area | manhua-translator | BallonsTranslator |
|------|-------------------|-------------------|
| **OCR** | PaddleOCR v5 (det + rec), Korean/Japanese/English/Chinese | Multiple detectors (RapidOCR, CRAFT, Surya, …) and OCRs (RapidOCR, EasyOCR, …) |
| **Translation** | PPIO GLM / Google Gemini | Google, DeepL, LLM API, chain, … |
| **Inpainting** | LaMa or OpenCV | LaMa, OpenCV, LDM, … |
| **Rendering** | Dynamic font sizing, style from image, stroke, CJK wrap | Font format, stroke, auto fit, style presets |
| **Crawler** | Scraper engine + site scrapers (ToonGod, MangaForFree, generic Playwright) | Manga/comic source (MangaDex, GOMANGA, Comick, Manhwa Reader, **Generic chapter URL**) |
| **Pipeline** | Image → OCR → Region grouping → Translation → Inpainting → Rendering | Same idea; batch queue, project-based |

---

## 1. Web Crawler (Built-in Comic Downloader)

### How manhua-translator does it

- **Base abstraction** (`scraper/base.py`):
  - `Manga` (id, title, url, cover_url), `Chapter` (id, title, url, index).
  - `BaseScraper(ABC)` with `search_manga`, `get_chapters`, `download_images`.
  - `ScraperConfig`: base_url, headless, timeouts, scroll/scroll_wait, http_mode, storage_state_path (cookies), rate_limit_rps, user_agent.

- **Downloader** (`scraper/downloader.py`):
  - `AsyncDownloader`: async download with **aiohttp**, semaphore-based concurrency, **retries with exponential backoff**, **rate limiter**.
  - `DownloadItem` (index, url, filename, referer) → `PageRecord` (index, url, path, ok, error).
  - Writes **manifest.json**: manga_id, chapter_id, created_at, pages (index, url, path, ok, error).

- **Engine** (`scraper/engine.py`):
  - `ScraperEngine(scraper, config)`: `search`, `list_chapters`, `download_chapter`, `download_manga`.
  - Output layout: `output_root / safe_name(manga) / safe_name(chapter) /` with images + manifest.

- **Site implementations**:
  - **ToonGod** / **MangaForFree**: Playwright-based (scroll, wait, extract image URLs), Cloudflare handling.
  - **generic_playwright**: Configurable selectors and scroll for arbitrary sites.

- **API** (FastAPI): `/scraper/search`, `/scraper/catalog`, `/scraper/chapters`, `/scraper/download` (async task), `/scraper/task/{id}`, state upload, cover proxy, access check.

### What we implemented in BallonsTranslator

- **Generic (chapter URL) source** in **Manga / Comic source**:
  - User pastes a **chapter page URL** (any site that serves images in HTML).
  - We **fetch HTML** with httpx, parse with BeautifulSoup, collect image URLs from `img[src]`, `img[data-src]`, and common `data-srcset`/lazy patterns.
  - **Download** images with retries and backoff, save as `001.png`, `002.png`, … (or original extension).
  - Write **manifest.json** in the same shape as manhua-translator (manga_id, chapter_id, pages with index/url/path/ok/error) so folders are compatible and traceable.
  - Rate limiting via existing `request_delay` in the manga source dialog.
  - **Open folder in BallonsTranslator** after download (same as other sources).

- **Limitation**: Sites that require JavaScript (e.g. Cloudflare, heavy SPA) are not supported by the HTTP-only fetcher; for those, manhua-translator’s Playwright-based stack or a browser extension remains the option. We document this in the doc and in the UI tooltip.

---

## 2. Masking / Inpainting

### manhua-translator

- **core/modules/inpainter.py**: LaMa or OpenCV; dilation (e.g. 12px); per-region `box_2d`.
- Skips inpainting for `[翻译失败]` and `[SFX:…]`; for `crosspage_role == "current_bottom"` extends box downward with padding.
- Writes debug artifacts (mask, inpainted image).

### BallonsTranslator

- Multiple inpainters (LaMa, LDM, OpenCV, etc.); mask from detector/OCR blocks, optional expansion.
- We did not change our masking model; the “skip failed/SFX” and crosspage ideas could be ported later if we add similar region metadata.

---

## 3. Text Box Control and Rendering

### manhua-translator

- **core/renderer.py**:
  - **StyleEstimator**: From original image crop (box) → text color (mode of text pixels), need_stroke (from mean brightness), **estimate_font_size** from box size and text length (line count estimate, padding_ratio, line_spacing).
  - **TextRenderer**: CJK-aware wrap (forbidden line start/end), **fit_text_to_box** (binary search font size so text fits), **fit_text_to_box_with_reference** (ref from first line or override, font_size_ref_range, fallback min/max).
  - **style_config**: font_size_ref_range, font_size_fallback, line_spacing_default, line_spacing_compact, etc. (YAML).
  - Renders with PIL; stroke when needed; centers text in box.

### BallonsTranslator

- **Font format / style**: Per-block font, size, stroke, alignment, “Auto fit to block” (scale to fit bbox).
- **EASYSCANLATE_INTEGRATION.md**: “Stroke outline outside only” option (stroke drawn only outside glyphs).
- We already have dynamic font sizing and fit-to-block; we did not port their exact StyleEstimator or YAML style config. Possible future improvement: estimate text color and “need stroke” from original region image.

---

## 4. Pipeline and Region Grouping

### manhua-translator

- **core/pipeline.py**: OCR → region grouping (merge nearby lines) → translation → inpainting → rendering.
- **core/crosspage_*.py**: Cross-page bubbles (split/carryover/pairing) for long strips.
- **OCR postprocessor**, **translation_splitter**, **quality_report**, **watermark_detector**.

### BallonsTranslator

- Detect → OCR → translate → inpaint → render; batch queue; project-based.
- Region merge tool and merge-by-distance (e.g. rapidocr_det merge_gap_px).
- We have translation context/glossary; we did not port crosspage splitting or watermark detection.

---

## 5. Summary of What We Implemented

- **Comic downloader (crawler)**:
  - **Generic (chapter URL)** source: paste URL → fetch HTML → extract image URLs → download with retry/backoff → write manifest.json (manhua-translator compatible) → optional “Open in BallonsTranslator”.
- **Documentation**: This file and tooltips so users know when to use Generic chapter URL vs other sources or external tools (Playwright/manhua-translator) for JS-heavy sites.

No changes to masking or rendering logic beyond what was already present; optional improvements (style estimation, crosspage, etc.) are noted for future work.
