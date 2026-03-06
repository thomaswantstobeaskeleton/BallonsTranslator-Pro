# BallonsTranslatorPro

**Repository:** [https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro](https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro) · **Version:** 1.7.0

Community fork of [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) with extended features. Original behavior and defaults are unchanged unless noted.

---

## At a glance

| Topic | Summary |
|-------|---------|
| **What** | Fork with 20+ detectors (incl. dual detection), 30+ OCR engines, 15+ inpainters, translation context & glossary, text eraser, batch queue, **Manga/Comic source** (MangaDex incl. raw/original language, GOMANGA, Manhwa Reader, Comick, Local folder), **batch export to PDF**, **duplicate/overlapping block check**, 370 fonts. Full docs and recommended settings. |
| **Upstream** | [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) — base project |
| **Merge** | Suitable for upstream as a separate experimental branch. See [CONTRIBUTING.md](CONTRIBUTING.md). |


**Recent additions:** **UI (Issue #7):** Title bar has **File** (open/save/export/import), **Edit** (with Keyword substitution submenu), **View** (grouped: panels, shortcuts, styles, Help), **Pipeline** (renamed from Run; stage toggles and presets), **Tools** (grouped: project, export, sources, queue, models). Left bar Open menu trimmed to open/save only; export/import are in File. Config → General split into Startup, Display, Typesetting, OCR result, Save, Canvas, Integrations. Canvas context menu: Edit has separators; **Run** (renamed from Detect & Run). **Secondary detector — outside bubbles only** (Config → Detector: use primary for bubbles e.g. YSGYolo, secondary e.g. EasyOCR for signs/captions only). **Ensemble (3+1) translator** now shows all translators in candidate/judge dropdowns with improved Zh→En defaults (Google, nllb200, LLM_API_Translator; judge LLM_API_Translator; use OpenRouter in LLM_API params). **Merge settings:** detector merge_gap_px (e.g. EasyOCR) and **Merge nearby blocks (collision)** in Config → DL Module to fix small text boxes. **Delete and Recover:** Deleting text blocks (canvas right-click → Delete and Recover) now syncs with the project so saved files and pipeline runs stay consistent; undo/redo restores or re-removes blocks in project pages. Manual mask/drawing edits are merged into the inpainted image before running Inpaint; per-block text eraser masks are applied when building the inpainting mask so holes are preserved; canvas refreshes after delete/recover to avoid display artifacts. **Google Translate** works without an API key by default (same as original BallonsTranslator); set an API key in Config → Translator → Google only if you need higher limits. **Translation context:** `data/translation_context/default` is created automatically when using series context so translation no longer fails with a missing directory. **Auto fit to box:** When Config → Typesetting is set to “Auto fit to box”, Detect/OCR/Translate runs now trigger auto font scaling for new blocks; you can also re-run via right-click → Format → **Auto fit font size to box** after changing font. **Context menu options** (right-click menu customization) dialog text clarified and grouped by category. **Inpainting with upscale:** When running with initial upscale but without re-running detection, block coordinates are scaled to the upscaled image so inpainting runs correctly. New docs: [TROUBLESHOOTING](docs/TROUBLESHOOTING.md), [STARRIVER](docs/STARRIVER.md), [CONJOINED_MODELS](docs/CONJOINED_MODELS.md), [DANGO_REFERENCE](docs/DANGO_REFERENCE.md). **Config:** Fresh installs use `config/config.example.json` as the template; your `config/config.json` is never overwritten by updates.

**Latest:** **Translator chain** — New "Chain" translator runs multiple translators in sequence (e.g. Japanese → English → Chinese); set `chain_translators` and `chain_intermediate_langs` in Config → Translator. **Open ACBF/CBZ** — File → Open → **Open ACBF/CBZ...** opens comic archives (`.cbz`/`.zip`); extracts to a folder and opens as project. **Batch translation script** — `python scripts/batch_translate.py --dir ./folder` runs detect/OCR/translate/inpaint from the command line; use `--no-detect`, `--no-ocr`, `--no-translate`, `--no-inpaint` to skip stages. **Auto fit font size to block** — Format panel checkbox "Auto fit to block" (and per-block option) scales font so text fits the bounding box when layout runs; more font size presets (e.g. 24, 52, 64, 80, 200) in the font size list. **Bug fixes:** Grok and other LLMs that return `{"1": "text"}`-style JSON are now accepted by LLM_API_Translator; pipeline no longer crashes when the text detector module failed to load (shows error and stops instead of `'NoneType' has no attribute 'detect'`).

---

### Disclaimer: models and testing

This fork adds **many new optional modules** (detectors, OCR engines, inpainters). **Not all of them have been tested in every environment** (Windows/Linux/macOS, CPU/CUDA, all language pairs). Some issues may persist—e.g. dependency conflicts (see [§8 Optional dependency conflicts](#8-optional-dependency-conflicts-and-workarounds)), OOM on low VRAM, or model-specific bugs. Use **docs/QUALITY_RANKINGS.md** and **docs/MANHUA_BEST_SETTINGS.md** for recommended combinations. If you hit a problem with a particular module, try another from the same category or report an issue with details (OS, device, config). Known dependency conflicts (e.g. craft_det, simple_lama) are documented in **docs/OPTIONAL_DEPENDENCIES.md**.

---

## Quick start

1. **Clone and run:** `git clone https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro.git && cd BallonsTranslator-Pro && python launch.py`
2. **First run:** Installs base deps and downloads default models into `data/`. If `config.json` is missing (e.g. fresh ZIP), the app loads **recommended defaults** from `config/config.example.json` and creates `config.json`. Transient connection errors (e.g. "Remote end closed connection") are retried automatically; if a download still fails, the log shows the path so you can download the file manually and restart.
3. **Config:** Open the settings panel → choose **Text detection**, **OCR**, **Inpainting**, **Translation** from the dropdowns
4. **New modules** appear automatically; install only the dependencies for the modules you use
5. **Updating:** Use **View → Help → Update from GitHub** to pull the latest changes without re-downloading; your config and local files are not overwritten. *This only works if you cloned the repo with git (e.g. `git clone ...`). If you downloaded a ZIP, download the latest ZIP from GitHub and replace the folder to update.* Optional: **Config → General → Auto update from GitHub on startup** (can cause issues — see tooltip).

### Portable / one-click setup (Section 11)

For a **portable-style** install (e.g. copy folder and run elsewhere):

1. **Setup script (optional)**  
   - **Windows:** run `setup.bat` to create a venv and install dependencies (or run `python launch.py` directly; it will install base deps and PyTorch).  
   - **Linux / macOS:** run `./setup.sh` (or `bash setup.sh`) for the same.

2. **Torch**  
   `launch.py` **auto-detects** GPU and installs the right PyTorch build (NVIDIA CUDA, AMD ROCm on Windows, or CPU). You can override with env **TORCH_COMMAND** (e.g. a custom `pip install torch ...` command) before the first run. Use `--reinstall-torch` to force reinstall.

3. **Fonts**  
   Pre-bundled fonts are in **`fonts/`**. Add `.ttf` / `.otf` / `.ttc` / `.pfb` there; they are loaded at startup. No need to move them for portable use.

4. **Models**  
   Default models download to **`data/`** (e.g. `data/models/`). To move or backup: copy the whole **`data/`** folder (and optionally **`config/`**) to the new location and point the app at the same project root (or keep the folder structure so `data/` and `config/` stay next to `launch.py`).

5. **Output**  
   Translated pages and project files are saved **inside the project folder** you open (File → Open Folder / Open Images). Export paths (Tools → Export all pages) are chosen in the dialog. No central “output” folder—each project is self-contained.

6. **Troubleshooting**  
   See **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** for GPU OOM, HuggingFace gated models, provider keys, and dependency conflicts.

---

## Key highlights (point by point)

- **Text detection:** 20+ detectors (CTD, Paddle, EasyOCR, YSGYolo, HF object-detection, MMOCR, Surya, Magi, CRAFT, DPText-DETR, etc.). **Box padding** (4–6 px) on CTD, Paddle, EasyOCR, YSGYolo, HF object-det, MMOCR, Surya to reduce clipped punctuation.
- **OCR:** 20+ engines (Paddle, manga_ocr, Surya, TrOCR, GOT-OCR2, Ocean, InternVL2/3, HunyuanOCR, etc.). **Crop padding** on many OCRs to avoid clipped text.
- **Inpainting:** lama_large_512px with **configurable mask dilation** (0–5); Simple LaMa, Diffusers (SD/SDXL/FLUX), LaMa ONNX, MAT, Fluently v4, etc.
- **Translation context:** Glossary, previous-page context, series-level storage, optional **context summarization** when near model limit (LLM_API_Translator).
- **UI:** Canvas right-click menu (30+ actions: copy/paste, merge, move up/down, spell check, trim, case change, gradient, text on path, detect in region, pipeline stages); **OCR auto-correct** (pyenchant, single-suggestion replacement after OCR); **text eraser** tool; **batch queue**; **Manga / Comic source** (MangaDex incl. raw/original language, GOMANGA, Manhwa Reader, Comick, Local folder — search and download where supported); **batch export to PDF** (Export all pages); **Check project** (missing files, invalid JSON, overlapping blocks); **keyboard shortcuts** (customizable); **keyword substitution** (OCR, pre-MT, post-MT).
- **Config panel:** Logical DPI, dark mode, display language, WebP lossless, typesetting defaults, dual text detection (primary + secondary detector).

**Models and fonts added in this fork:** The fork ships with **many more** detection, OCR, and inpainting modules than the original (15+ text detectors, 30+ OCR engines, 15+ inpainters), plus **370+ fonts** in the `fonts/` folder (included in the repo for accessibility and built-in options; supported extensions: `.ttf`, `.otf`, `.ttc`, `.pfb`). Custom fonts in `fonts/` are loaded at startup. Config → Typesetting: **Show only custom fonts** limits the font list to those. Not every module is tested in every environment—see [Disclaimer: models and testing](#disclaimer-models-and-testing) above.

### Recommended settings (manhua / Chinese comics)

For **manhua** (Chinese comics), see [docs/MANHUA_BEST_SETTINGS.md](docs/MANHUA_BEST_SETTINGS.md). Quick reference:

| Stage | Recommended | Key settings |
|-------|-------------|--------------|
| **Detection** | CTD | detect_size 1280, box score 0.42–0.48, box_padding 4–6 |
| **OCR** | Surya OCR | Language: Chinese (Simplified), Fix Latin misread: True, crop_padding 6–8 |
| **Inpainting** | lama_large_512px | mask_dilation 1–2, inpaint_size 1024 |

For quality rankings (tier-based), see [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md).

---

## Tutorials (step-by-step)

### Translation context and glossary (LLM_API_Translator)

**Goal:** Keep terminology and style consistent across pages and chapters (e.g. cultivation manhua).

1. **Open a project** (File → Open Folder or Open Images).
2. **Edit → Translation context (project)...** (or Config → DL Module → Translator → **Translation context (project)...**).
3. **Series context path:** Enter a folder path or series ID (e.g. `urban_immortal_cultivator`). Resolves to `data/translation_context/<id>/`.
4. **Project glossary:** Add entries, one per line: `source -> target` or `source = target`. Example: `丹田 -> dantian`.
5. **Save** — writes to project JSON.
6. **Translator params** (Config → Translator): Set **context_previous_pages** (0–5), **context_max_chars** (e.g. 2000), **context_trim_mode** (`full` = more lines per page; `compact` = at most 2 lines per page to save tokens), **summarize_context_when_over_limit** (optional). **translation_glossary** in params merges with project and series glossary.
7. **Run** — pipeline passes previous pages and glossary to the LLM; after each page, appends to `recent_context.json` in the series folder for cross-chapter consistency.

**Glossary format:** One entry per line: `source -> target` or `source = target`. Lines starting with `#` are ignored. First occurrence of each source term wins when merging glossaries.

**Files:** `data/translation_context/<series_id>/glossary.txt`, `recent_context.json`. See [§12.8](#128-translation-context-glossary-series-previous-pages-summarization).

**Proxy (LLM / API modules):** If behind a proxy, set **Proxy** in Config → Translator (LLM_API_Translator) or the relevant module params. Format: `http://host:port` or `socks5://host:port`. URLs without scheme (e.g. `127.0.0.1:7897`) are auto-normalized to `http://...` via `utils/proxy_utils.py`.

---

### OCR spell check / auto-correct (after OCR)

**Goal:** Automatically fix common OCR misspellings (e.g. "teh" → "the") right after OCR runs.

1. **Config → General → OCR result** → check **Spell check / Auto-correct OCR result**. Stored in `pcfg.ocr_spell_check`.
2. **Requires:** [pyenchant](https://pyenchant.github.io/pyenchant/) and a system dictionary (e.g. `en_US`). Install: `pip install pyenchant`. On Windows, you may need to install [Enchant](https://github.com/AbiWord/enchant) or use a compatible spell library.
3. **How it works:** After each OCR run, `spell_check_textblocks` (in `utils/ocr_spellcheck.py`) runs as an OCR postprocess hook. For each word: if it's misspelled and there is **exactly one suggestion**, the word is replaced. If there are 0 or 2+ suggestions, the word is left unchanged (avoids wrong corrections).
4. **Manual spell check:** You can also run spell check manually via the canvas right-click menu → **Spell check source text** or **Spell check translation** on selected blocks (same logic, requires pyenchant).

---

### Dual text detection (primary + secondary)

**Goal:** Catch regions missed by one detector (e.g. captions vs speech bubbles).

1. **Config → Text detection** → check **Run second detector (dual detect)**.
2. **Secondary:** dropdown — select a different detector (e.g. CTD + Surya).
3. **Secondary detector params** — appears below; set device, thresholds, etc. for the secondary.
4. **Run** — pipeline runs primary, then secondary, merges blocks by IoU threshold 0.4 and masks when same shape.
5. Progress: **Detecting (dual: primary + secondary):** when active.

**Recommended dual combo (Paddle v5 + HF ogkalu):** Use **paddle_det_v5** as primary (strong on captions, dense text, vertical) and **hf_object_det** (ogkalu/comic-text-and-bubble-detector) as secondary to add bubble regions Paddle may miss. In Secondary detector params, set **labels_include** to `bubble,text_bubble` so HF adds only speech/thought bubbles; add `text_free` if you want HF to also catch SFX/captions Paddle missed. **box_padding** 4–6 on both detectors.

---

### HF object detector (ogkalu) and labels_include

**Goal:** Use the Hugging Face object detector with the default comic model and choose which text regions to detect (bubbles only, or bubbles + SFX/captions).

1. **Install:** `pip install transformers torch`.
2. **Config → Text detection** → select **hf_object_det** or **rtdetr_comic**. **rtdetr_comic** defaults to the comic model with **no model id** (leave **model_id** empty). **hf_object_det** default **model_id** is `ogkalu/comic-text-and-bubble-detector` (RT-DETR fine-tuned for comic text and bubbles).
3. **labels_include** — Comma-separated list of which classes to keep. The ogkalu model outputs three classes:
   - **bubble** — Speech/thought bubble shape (the drawn balloon).
   - **text_bubble** — Text inside a bubble.
   - **text_free** — Text not in a bubble: sound effects (SFX), captions, signs, narrative.
   - **Examples:** `bubble,text_bubble` = detect only bubbles (recommended when using HF as secondary in dual detection). `bubble,text_bubble,text_free` = detect everything (default). Leave empty to keep all classes.
4. **score_threshold** (e.g. 0.35–0.4) — Min confidence for bubble/text_bubble. **score_threshold_text_free** — Threshold for text_free (often lower, e.g. 0.15–0.2). **box_padding** — 4–6 px recommended.

**Inpainting with HF object detector:** The **ogkalu/comic-text-and-bubble-detector** model outputs separate boxes for bubbles and text (e.g. `bubble`, `text_bubble`). Overlapping and nested boxes are merged (by IoU and containment) so each speech bubble is inpainted once. Use **merge_overlap_iou** (default 0.35) to control merging; lower = merge more, 1.0 = no merge.

See [Dual text detection](#dual-text-detection-primary--secondary) for using HF as secondary with Paddle v5 and suggested **labels_include** values.

---

### Manga / Comic source (multiple sources)

**Goal:** Search, list chapters, and download manga/manhua/manhwa from several sources; or open a local folder of images as a project.

**Tools → Manga / Comic source...** opens the dialog. Choose a **Source** from the dropdown:

| Source | Search | Chapter list | Download | Notes |
|--------|--------|--------------|----------|--------|
| **MangaDex** | By title | By language | Yes (001/002… pages) | Official API; data-saver option |
| **MangaDex (raw / original language)** | By title, filter by original language | Raw language (ja/ko/zh…) | Yes | Find untranslated manga; download chapters in original language to translate |
| **MangaDex (by chapter URL)** | — | Paste chapter URL | Yes | Single chapter by link |
| **Comick** | By title (Comick Source API) | Yes | No | Many aggregator sites; search merges all sources (NDJSON); chapter links only |
| **GOMANGA** | By title | Yes | Yes | Unofficial API; direct image URLs; shows clear error if upstream returns 403 |
| **Manhwa Reader** | Filter /api/all by title | Yes | Yes | Manhwa/webtoon; shown in list only when API is up |
| **Local folder** | — | — | — | Open folder of images as project |

#### Search and load chapters (MangaDex, Comick, GOMANGA, Manhwa Reader)

1. **Search by title:** Enter manga name → **Search** → results show title. Select a result → **Load chapters**.
2. **MangaDex only — by URL:** Paste a MangaDex chapter URL (e.g. `https://mangadex.org/chapter/abc123...`) → **Load chapter** — fetches that chapter directly.
3. **Language:** Dropdown for chapter feed (e.g. English, Japanese, Chinese Simplified). For **MangaDex (raw / original language)** this becomes **Raw language (chapters to load)** — use it to search and load chapters in the original language (e.g. Japanese, Korean, Chinese) for translating. Stored in `manga_source_lang`. (MangaDex only; other sources ignore.)
4. **Quality:** **Use data-saver** — smaller images, faster download (MangaDex only). Stored in `manga_source_data_saver`.

#### Download

5. **Download folder:** Choose base folder. Default: `~/BallonsTranslator/Downloaded Chapters` (created automatically). Stored in `manga_source_download_dir`.
6. **Request delay:** 0–2 s between API requests (rate limiting). Stored in `manga_source_request_delay`.
7. Select chapters (checkboxes) → **Download selected chapters**.
8. **Page naming:** Pages are saved as **001.png**, **002.png**, … (or original ext) so BallonsTranslator loads them in reading order. Original MangaDex filenames are not used.
9. **Folder structure:** `{download_folder}/{manga_title}/{chapter_display}/` (e.g. `Ch.1 – Vol.1`).

#### After download

10. **Open in BallonsTranslator after download** — when checked, first chapter folder opens automatically. Stored in `manga_source_open_after_download`.
11. **Open folder in BallonsTranslator** — button to manually open the first downloaded chapter folder as a project.

---

### Batch queue

**Goal:** Process multiple folders in sequence (e.g. chapters 1–10).

1. **Tools → Batch queue...**
2. **Add folder(s)...** — add one or more folders.
3. **Add folder (include subfolders)** — add selected folder and each immediate subfolder as separate items.
4. **Start queue** — processes folders one by one (same as headless `--exec_dirs`).
5. **Pause** / **Resume** — halt or continue at page boundaries.
6. **Cancel queue** — stop and clear remaining queue.

---

### Canvas right-click menu (full reference)

In **text edit mode**, right-click on the canvas to open a context menu. Items are enabled/disabled based on selection state.

| Action | Shortcut | When enabled | Description |
|--------|----------|--------------|-------------|
| **Copy** | Ctrl+C | Always | Copy source text to clipboard |
| **Paste** | Ctrl+V | Always | Paste from clipboard |
| **Copy translation** | — | Always | Copy translation of selected blocks (one per line) |
| **Paste translation** | — | Always | Paste lines into translation fields of selected blocks |
| **Delete** | Ctrl+D | Always | Delete selected blocks |
| **Copy source text** | Ctrl+Shift+C | Always | Copy source text |
| **Paste source text** | Ctrl+Shift+V | Always | Paste source text |
| **Delete and Recover removed text** | Ctrl+Shift+D | Always | Delete blocks but keep text in recover list |
| **Clear source text** | — | Always | Clear source text of selected blocks |
| **Clear translation** | — | Always | Clear translation of selected blocks |
| **Select all** | Ctrl+A | Always | Select all blocks on page |
| **Spell check source text** | — | ≥1 selected | Auto-correct source text (pyenchant) |
| **Spell check translation** | — | ≥1 selected | Auto-correct translation (pyenchant) |
| **Trim whitespace** | — | ≥1 selected | Remove leading/trailing whitespace |
| **To uppercase** / **To lowercase** | — | ≥1 selected | Change case |
| **Toggle strikethrough** | — | ≥1 selected | Toggle strikethrough |
| **Gradient type** → Linear / Radial | — | ≥1 selected | Set gradient |
| **Text on path** → None / Circular / Arc | — | ≥1 selected | Draw text along path |
| **Merge selected blocks** | — | ≥2 selected | Merge into one block |
| **Split selected region(s)** | — | ≥1 selected | Split block(s) |
| **Move block(s) up** | — | 1 selected, not first | Swap with block above |
| **Move block(s) down** | — | 1 selected, not last | Swap with block below |
| **Apply font formatting** | — | Always | Apply format panel to selected |
| **Auto layout** | — | Always | Auto-split lines to fit region |
| **Reset Angle** | — | Always | Reset block angle |
| **Squeeze** | — | Always | Squeeze text layout |
| **Detect text in region** | — | After right-drag rect | Run detector in drawn region only |
| **Detect text on page** | — | Always | Run detector on full page |
| **Translate** | — | Always | Translate selected blocks |
| **OCR** | — | Always | Run OCR on selected blocks |
| **OCR and translate** | — | Always | OCR then translate |
| **OCR, translate and inpaint** | — | Always | Full pipeline |
| **Inpaint** | — | Always | Inpaint only |

**Detect text in region:** Right-drag a rectangle first (rubber-band); then right-click. The menu shows **Detect text in region** only when a rect was drawn. New blocks are appended in full-image coordinates; no OCR or translation runs.

---

### Detect text in region

**Goal:** Add text blocks in one area without re-running the full pipeline.

1. **Text edit mode** — ensure you're in text edit mode (T).
2. **Right-drag** a rectangle on the canvas (rubber-band).
3. **Right-click** → **Detect text in region**.
4. New blocks are appended in full-image coordinates. No OCR or translation.
5. **Or:** Right-click without a region → **Detect text on page** — runs detector on full page.

---

### Inpaint tiling (OOM fix)

**Goal:** Avoid OOM when inpainting large images.

1. **Config → Inpainting** (or DL Module → Inpainter params).
2. **Inpaint tile size:** Set 512 or 1024 (0 = off).
3. **Inpaint tile overlap:** 64 px typical.
4. Tiling activates when image is > 1.5× tile size. Tiles are processed separately and stitched.

---

### Region merge tool

**Goal:** Merge overlapping or adjacent text blocks with configurable rules.

1. **Edit → Region merge tool** (or Ctrl+Shift+M) — opens settings dialog.
2. **Merge mode:** Vertical, Horizontal, Vertical then horizontal, Horizontal then vertical, None.
3. **Text merge order:** LTR, RTL, TTB labels (e.g. `balloon,qipao,shuqing` for RTL).
4. **Label merge strategy:** Prefer shorter, Use first, Combine, Prefer non-default.
5. **Blacklist labels:** Exclude from merge (e.g. `other`).
6. **Require same label** or **Merge only within specific label groups**.
7. **Geometry:** Max vertical/horizontal gap (px; 0 = must touch; negative = require overlap), min overlap ratio (%). Strict mode: gaps 0 or negative, overlap 98–100%.
8. **Merge result type:** Axis-aligned rectangle or Rotated rectangle.
9. **Run on current page** or **Run on all pages**.

**Settings persistence:** Region merge settings are saved to `pcfg.region_merge_settings` when you run; they are restored when you reopen the dialog.

---

### Auto region merge after run

**Goal:** Automatically run the Region merge tool after each pipeline run (detect/OCR/translate/inpaint).

1. **Config → General → Startup** → **After Run: Region merge** dropdown.
2. **Never** (default) — no auto merge.
3. **After run: on all pages** — runs Region merge on every page in the project using saved settings.
4. **After run: on current page only** — runs Region merge only on the current page.
5. Uses settings from **Edit → Region merge tool** (merge mode, labels, geometry, etc.). Configure that dialog first; settings are persisted in `region_merge_settings`.

---

### Text eraser tool

**Goal:** Erase parts of text blocks by painting over them (mask-based).

1. **Drawing mode** — switch to drawing board (P).
2. **Text eraser** tool — select from the drawing tools bar (eraser icon).
3. **Left-click and drag** over the text to erase. Stroke modifies each block's mask.
4. **Right-click** does not start a stroke (avoids accidental lines).
5. **Ctrl+Z** — undo the last eraser stroke.

---

### Text formatting (gradient, text on path, warp)

**Goal:** Style text for balloons and SFX (sound effects).

1. **Select one or more text blocks** — right-hand format panel appears.
2. **Gradient type:** Format panel → **Gradient** group → Linear or Radial. Or right-click canvas → **Gradient type** → Linear / Radial.
3. **Text on path** ([#1138](https://github.com/dmMaze/BallonsTranslator/issues/1138)): Format panel → **Text on path** dropdown: None, Circular, or Arc. **Circular** — text along a full circle. **Arc** — text along an arc; set **Arc degrees** (30–360°, default 180°) for the arc span.
4. **Warp** ([#1093](https://github.com/dmMaze/BallonsTranslator/issues/1093)): Format panel → **Warp** dropdown: None, Arc, Arch, Bulge, Flag. **Warp strength** (0.1–1) for distortion intensity.

---

### Manage models

**Goal:** Check which models are downloaded and download missing ones.

1. **Tools → Manage models...**
2. Table lists all modules (detection, OCR, inpainting) with status: downloaded / missing / hash mismatch.
3. **Check** — refresh status (optionally include import check).
4. Select rows → **Download selected** — downloads missing models to `data/models/`.
5. Models are stored per module (e.g. `comictextdetector.pt.onnx`, `lama_large_512px.ckpt`).

---

### Run presets and pipeline stages

**Goal:** Run only specific stages (e.g. detect + OCR without translate).

1. **Run** menu (title bar) → **Pipeline presets**:
   - **Full** — detect, OCR, translate, inpaint (default).
   - **Detect + OCR only** — no translation or inpainting.
   - **Translate only** — assumes detection/OCR already done.
   - **Inpaint only** — assumes text already placed.
2. **Enable Text Detection / OCR / Translation / Inpainting** — checkboxes to enable/disable each stage for the next run.
3. **Run without update textstyle** — runs pipeline but does not apply global font format to new blocks.

---

### Export and import (File menu)

**Goal:** Export text for external editing or import translations from files.

1. **File** (left bar Open button) → **Export as Doc** — exports project to `.docx` (Word).
2. **Import from Doc** — imports from `.docx` or `.json` project file.
3. **Export source text as TXT** / **Export translation as TXT** — plain text, one block per line.
4. **Export source text as markdown** / **Export translation as markdown** — markdown format.
5. **Import translation from TXT/markdown** — paste translations from a file; lines map to blocks in order.

**Tools → Export all pages** — exports result images for all pages to a folder as **001.ext**, **002.ext**, **003.ext**, … (in project page order), so the exported folder uses the same naming convention as manga chapter downloads and natural sort. Respects Config → Save format, quality, WebP lossless. Option **Also create PDF from exported images** builds a single `exported.pdf` in that folder (requires `pip install img2pdf`).

---

### File menu quick reference (left bar Open button)

| Item | Shortcut | Description |
|------|----------|-------------|
| **Open Folder** | Ctrl+O | Open folder as project |
| **Open Images** | — | Select image files; opens as project |
| **Open Project** | — | Open `.json` project file |
| **Open Recent** | — | Submenu of recent projects |
| **Save Project** | Ctrl+S | Save project JSON |
| **Export as Doc** | — | Export to `.docx` |
| **Import from Doc** | — | Import from `.docx` or `.json` |
| **Export source/translation as TXT** | — | Plain text |
| **Export source/translation as markdown** | — | Markdown |
| **Import translation from TXT/markdown** | — | Paste translations from file |

---

### Edit menu quick reference

| Item | Description |
|------|-------------|
| **Undo** / **Redo** | Ctrl+Z / Ctrl+Shift+Z |
| **Search** | Current page search (Ctrl+F) |
| **Global Search** | All pages search (Ctrl+G) |
| **Keyword substitution for source text** | Replace in OCR output |
| **Keyword substitution for machine translation source text** | Replace before translation |
| **Keyword substitution for machine translation** | Replace after translation |
| **Translation context (project)...** | Project glossary, series path |

---

### View menu quick reference

| Item | Description |
|------|-------------|
| **Display Language** | UI language (submenu) |
| **Drawing Board** | Switch to drawing mode (P) |
| **Text Editor** | Switch to text edit mode (T) |
| **Keyboard Shortcuts...** | Customize keybinds (Ctrl+K) |
| **Context menu options...** | Show/hide canvas right-click actions by category (Ctrl+Shift+O) |
| **Help** | **Documentation** (open README), **About** (version), **Update from GitHub** (pull latest changes; config preserved) |
| **Dark Mode** | Toggle dark theme |

---

### Go menu and page navigation

| Item | Shortcut | Description |
|------|----------|-------------|
| **Previous page** | PgUp | Go to previous page |
| **Next page** | PgDown | Go to next page |
| **Previous (alt)** | A | Alternate key |
| **Next (alt)** | D | Alternate key |

**Page list** (left bar: toggle **Show page list**): Thumbnail list of all pages. Right-click a page for:
- **Reveal in File Explorer** — open folder and select the image file
- **Translate selected images** — run translation on selected pages only
- **Remove from Project** — remove page from project (does not delete the image file)

---

### Tools menu quick reference

| Item | Description |
|------|-------------|
| **Region merge tool** | Merge overlapping/adjacent blocks (Ctrl+Shift+M) |
| **Re-run detection only** | Run detector on all pages; keep existing OCR/translation |
| **Re-run OCR only** | Run OCR on all pages; keep existing detection/translation |
| **Export all pages** | Export result images to a folder as 001.ext, 002.ext, … (page order); optional **Also create PDF** (img2pdf) |
| **Check project** | Validate project: missing images, invalid JSON, duplicate/overlapping text blocks |
| **Manga / Comic source...** | Search and download from MangaDex (incl. raw/original language), GOMANGA, Manhwa Reader; Comick (search/list only); Local folder |
| **Batch queue...** | Process multiple folders in sequence |
| **Manage models...** | Check/download detection, OCR, inpainting models |

---

### Search (current page vs global)

**Goal:** Find text across the project.

1. **Edit → Search** (Ctrl+F) — search **current page** only. Highlights matches in source/translation; supports case-sensitive, whole word, regex (Config → General).
2. **Edit → Global Search** (Ctrl+G) — search **all pages** in the project. Opens a panel with a tree of matches; click to jump to the block.
3. **Left bar:** Toggle **Global Search** checkbox to show/hide the global search panel.

---

### Keyword substitution (Edit menu)

**Goal:** Replace keywords in source text, pre-MT source, or translation (e.g. fix OCR errors, standardize terms).

1. **Edit → Keyword substitution for source text** — applies to OCR output (source text). Stored in `pcfg.ocr_sublist`.
2. **Edit → Keyword substitution for machine translation source text** — applies before translation (pre-MT). Stored in `pcfg.pre_mt_sublist`.
3. **Edit → Keyword substitution for machine translation** — applies after translation. Stored in `pcfg.mt_sublist`.
4. Each dialog: table with **Keyword**, **Substitution**, **Use regex**, **Case sensitive**. **New** / **Delete** to add/remove rows.
5. Substitutions run automatically during pipeline (OCR stage, pre-translate, post-translate). Order of rows matters (first match wins when overlapping).

---

### Project folder structure

| Path | Purpose |
|------|---------|
| `data/models/` | Detection, OCR, inpainting model files (e.g. `comictextdetector.pt.onnx`, `lama_large_512px.ckpt`) |
| `data/translation_context/<series_id>/` | Series glossary (`glossary.txt`), recent context (`recent_context.json`) |
| `data/libs/` | Some runtime libraries |
| `config/config.json` | User config (modules, shortcuts, save format, etc.) |
| `config/config.example.json` | Recommended defaults (used on first run when config.json is missing) |
| `fonts/` | Custom fonts (370+ included) |
| `~/BallonsTranslator/Downloaded Chapters` | Default Manga source download folder |

**Project JSON** (`imgtrans_<folder_name>.json` in project folder): `pages` (page name → list of TextBlock), `image_info` (per-page finish_code, etc.), `current_img`, `translation_glossary`, `series_context_path`. Subfolders: `mask/`, `inpainted/`, `result/` for masks, inpainted images, and final output.

---

### Text style presets

**Goal:** Save and reuse font formats (font, size, color, stroke, etc.) across blocks.

1. **Format panel** (right side) — when a block is selected, use **Save as default** to write current format to global config, or **Apply to all blocks** to apply current format to every block.
2. **Text style preset panel** — shows saved presets (Config → General: **Show text style preset**). Click a preset to apply it to the selected block; double-click to edit the name.
3. **View → Import Text Styles** / **Export Text Styles** — load or save presets to a JSON file. Stored in `pcfg.text_styles_path` (default: `data/text_styles/default.json`).
4. **Independent text styles per project** (Config → Typesetting) — when enabled, each project can have its own preset set.

---

### Saladict and text selection mini menu

**Goal:** Look up selected text in Saladict (dictionary) or search engine.

1. **Config → General → Saladict / search** — **Show mini menu when selecting text** (`textselect_mini_menu`). When enabled, selecting text in a block shows a mini menu.
2. **Saladict shortcut** — default Alt+S. When you select text and trigger this shortcut, the app copies the selection and sends the shortcut to Saladict (must be installed; see [doc/saladict.md](doc/saladict.md)).
3. **Search engine URL** — for "Search" in the mini menu; opens `search_url` + selected text (e.g. Google search).

---

## Table of contents

- [Contributing & guidelines](CONTRIBUTING.md) — Git practices, community etiquette, upstream merge
- [Disclaimer: models and testing](#disclaimer-models-and-testing) — Not all modules tested; issues may persist
- [Tutorials (step-by-step)](#tutorials-step-by-step) — Translation context, dual detection, **HF object detector (labels_include)**, Manga source, batch queue, canvas right-click, detect in region, OCR spell check, auto region merge, text eraser, text formatting, manage models, run presets, export/import, Tools menu, keyword substitution, project structure, text style presets, Saladict
1. [Summary of modifications](#1-summary-of-modifications)
2. [How to run the application](#2-how-to-run-the-application)
3. [Text detection – all modules and how to run](#3-text-detection--all-modules-and-how-to-run)
4. [OCR – all modules and how to run](#4-ocr--all-modules-and-how-to-run)
5. [Inpainting – all modules and how to run](#5-inpainting--all-modules-and-how-to-run)
6. [Translation modules](#6-translation-modules)
7. [Settings reference: inpaint size, mask dilation, detection, OCR, formatting](#7-settings-reference-inpaint-size-mask-dilation-detection-ocr-formatting) — incl. §7.8 config reference, §7.9 shortcuts, §7.10 drawing tools
8. [Optional dependency conflicts and workarounds](#8-optional-dependency-conflicts-and-workarounds)
9. [New and modified files](#9-new-and-modified-files)
10. [Fixes and behavior changes](#10-fixes-and-behavior-changes)
11. [Documentation and references](#11-documentation-and-references)
12. [UI, workflow, and export enhancements](#12-ui-workflow-and-export-enhancements)  
    - [12.6 Batch processing queue (#1020)](#126-batch-processing-queue-1020)  
    - [12.7 Translation context: glossary, series, previous pages, summarization](#127-translation-context-glossary-series-previous-pages-summarization)  
    - [12.8 Translation context: glossary, series, previous pages, summarization (detail)](#128-translation-context-glossary-series-previous-pages-summarization)  
    - [12.9 Dual text detection (primary + secondary detector)](#129-dual-text-detection-primary--secondary-detector)  
    - [12.10 Text eraser tool](#1210-text-eraser-tool)  
    - [12.11 Text edit panel and right panel layout](#1211-text-edit-panel-and-right-panel-layout)

---

## 1. Summary of modifications

This fork adds **many new optional modules** and applies **fixes and setting improvements**. Original behavior and defaults are unchanged unless noted. New modules are discovered automatically via the existing registry (no changes to core launch or config flow). You only install extra dependencies for the modules you use. **Not all added models have been tested in every environment**—see [Disclaimer: models and testing](#disclaimer-models-and-testing).

### Text detection
- MMOCR, PP-OCRv5, Surya, Magi (Manga Whisperer), TextMamba (stub), CRAFT (standalone), **rtdetr_comic** (RT-DETRv2 comic text & bubble, no model id required), HF object-detection (default: ogkalu comic-text-and-bubble-detector), DPText-DETR, SwinTextSpotter v2
- **Box padding** (4–6 px recommended) on CTD, Paddle, EasyOCR, YSGYolo, HF object-det, MMOCR, Surya — reduces clipped punctuation (?, !) and character edges

### OCR
- 20+ new backends: **PP-OCRv5** (paddle_rec_v5), **PaddleOCR-VL** (paddle_vl / PaddleOCRVLManga), TrOCR, GOT-OCR2, GLM-OCR, Donut, PaddleOCR-VL (HF), Qwen2-VL 7B, DeepSeek-OCR, LightOn, Chandra, DocOwl2, Nanonets, Ocean-OCR, InternVL2/3, Florence-2, MiniCPM-o, OCRFlux, HunyuanOCR, Manga OCR Mobile (TFLite), Nemotron Parse (full-page)
- **Crop padding** on many OCRs (0–24 px) to reduce clipped text at edges

### Inpainting
- Simple LaMa, Diffusers (SD 1.5, SD2 768, SDXL 1024, DreamShaper, FLUX Fill, Kandinsky), RePaint, LaMa ONNX (general + manga), Qwen-Image-Edit, MAT, CUHK Manga, Fluently v4
- **Mask dilation** configurable (0–5) for lama_large_512px; **inpaint size** options per inpainter
### Translation
- **Translation context and glossary:** Cross-page and cross-chapter terminology consistency via project/series glossary, previous-page context, optional context summarization when near model limit. See [§6.1](#61-translation-context-and-glossary) and [§12.8](#128-translation-context-glossary-series-previous-pages-summarization).

### Settings and fixes
- CTD: box score threshold, merge tolerance, min box area, custom ONNX path
- **Dual text detection:** Run a second detector and merge results. See [§12.9](#129-dual-text-detection-primary--secondary-detector).
- Config panel: Logical DPI, display language, dark mode, font scale, recent projects limit, confirm before Run, OCR spell-check, typesetting defaults, WebP lossless, default device, unload after idle — all persisted in config
### UI and workflow (point by point)
- **Canvas right-click:** Categorized menu (Edit, Text, Block, Image/Overlay, Transform, Order, Format, Detect & Run); **Create text box** (right-click for default size, or right-drag to set size; Block → Create text box or **Ctrl+Shift+N** at cursor); **Configure menu...** to show/hide actions (also View → Context menu options, **Ctrl+Shift+O**); Detect text in region (right-drag rect) and on page; Merge selected blocks; Move block(s) up/down; Copy/Paste translation; Clear source/translation; Select all
- **Spell check:** Spell check source text and translation (pyenchant)
- **Text formatting:** Trim whitespace; To uppercase / To lowercase; Gradient type (Linear/Radial); Text on path (None/Circular/Arc, [#1138](https://github.com/dmMaze/BallonsTranslator/issues/1138))
- **Text eraser tool:** In drawing mode, erase parts of text blocks by painting over them (mask-based; undo with Ctrl+Z). See [§12.10](#1210-text-eraser-tool)
- **Text edit panel:** Default width closer to minimum; less cramped layout. See [§12.11](#1211-text-edit-panel-and-right-panel-layout)
- **Translation context (project):** Edit menu and Translator config open a dialog to set project series path and project glossary for cross-chapter consistency
- **Lossless WebP** (#1055): Config → Save when format is WebP; Save and Export all pages respect it
- **Batch export to PDF:** In **Tools → Export all pages**, checkbox **Also create PDF from exported images** builds `exported.pdf` in the chosen folder (requires `pip install img2pdf`).
- **Check project:** **Tools → Check project** validates missing image files, invalid project JSON, and **duplicate/overlapping text blocks** (reports page name and block index pairs).
- **Manga / Comic source** (Tools menu): Multiple sources — **MangaDex** (search, chapter URL, download), **MangaDex (raw / original language)** (search by original language, download raw chapters to translate), **GOMANGA** (search, chapters, download), **Manhwa Reader** (manhwa/webtoon, shown only when API is up; search, chapters, download), **Comick** (search and chapter list only; no download), **Local folder** (open folder of images as project). List chapters by language (MangaDex); download as 001/002… pages where supported; optional open folder in app; config persistence and rate limiting.
- **Batch queue** (Tools → Batch queue...): Process multiple folders in sequence with Pause/Resume/Cancel ([#1020](https://github.com/dmMaze/BallonsTranslator/issues/1020))
- **Default download folder:** `~/BallonsTranslator/Downloaded Chapters` (created automatically when empty)
- **Keyboard shortcuts:** View → Keyboard Shortcuts (Ctrl+K) to view and customize keybinds; shortcuts persist in config
- **Typesetting:** Auto-adjust text size to fit text box (Config → General: Font Size "decide by program" + Auto layout); font size list includes 30, 32, 34, 40, 44 for finer control
- **Drag-and-drop:** Folder or images onto the canvas to open a project; copy-paste (File → Open) also works

### Documentation
- **Troubleshooting:** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — GPU OOM, HuggingFace gated models, provider keys, dependency conflicts
- **Recommended settings:** [docs/MANHUA_BEST_SETTINGS.md](docs/MANHUA_BEST_SETTINGS.md) — detection, OCR, inpainting for manhua (Chinese comics)
- **Quality rankings:** [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md), [docs/BEST_MODELS_RESEARCH.md](docs/BEST_MODELS_RESEARCH.md), [docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md)
- **Optional deps:** [docs/OPTIONAL_DEPENDENCIES.md](docs/OPTIONAL_DEPENDENCIES.md), [docs/INSTALL_EXTRA_DETECTORS.md](docs/INSTALL_EXTRA_DETECTORS.md)
- **Translation context:** [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md)

### Upstream issues and PRs (dmMaze/BallonsTranslator)

This fork implements or aligns with several upstream feature requests:

- **[#126](https://github.com/dmMaze/BallonsTranslator/issues/126):** Eraser → **Text eraser** tool; **Help** menu (View → Help); **opacity** toggles; **natural sort**; **Escape** (text eraser + rect cancel); **model download** scripts: Linux (`scripts/download_models.sh`), **macOS** (`scripts/download_models_macos.sh`); **Export as** (Tools → Export all pages as…, File → Export current page as…) with format PNG/JPEG/WebP/JXL; **OpenCV-only inpainters** `opencv-tela` and `opencv-telea` (no model download). .kra/PSD export and optional detector–cleaner module not implemented.
- **[#41](https://github.com/dmMaze/BallonsTranslator/issues/41):** **MangaInpainter** → **cuhk_manga_inpaint** (see [doc/INSTALL_CUHK_MANGA.md](doc/INSTALL_CUHK_MANGA.md)).
- **[#35](https://github.com/dmMaze/BallonsTranslator/issues/35):** Text **opacity**; **shapes**: rect tool has **Ellipse** option; **per-letter stroke color** for text-on-path (`stroke_rgb_per_char`); font menu **long names** (min width 220px, 20 visible items).
- **PR [#991](https://github.com/dmMaze/BallonsTranslator/pull/991):** **Translate selected images** in page list context menu.
- **PR [#974](https://github.com/dmMaze/BallonsTranslator/pull/974):** **Spell check panel** (View → Spell check panel): Source/Translation, Check current page, list + Replace with suggestion (pyenchant). Context menu and OCR auto-correct unchanged.
- **PR [#1105](https://github.com/dmMaze/BallonsTranslator/pull/1105):** Preset **warp** and **text eraser**; Escape. Interactive mesh/quad warp not implemented.
- **PR [#1070](https://github.com/dmMaze/BallonsTranslator/pull/1070) (Image overlay):** Not implemented.

---

## 2. How to run the application

### Base setup (same as original)

- **Python:** 3.10 or 3.11 (≤ 3.12; avoid Microsoft Store Python).
- **Git:** Installed and in PATH.

```bash
# Clone the repository
git clone https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro.git
cd BallonsTranslator-Pro

# First run: installs base deps and downloads default models into data/
python launch.py

# Update dependencies and code
python launch.py --update
```

If model downloads fail, the app retries transient connection errors automatically. If a file still fails, the error message shows the path — download the file from the URL in the log and save it to that path, then restart. You can also use the original README links (MEGA / Google Drive) and place the `data` folder in the project root.

### Running

- **GUI:**  
  `python launch.py`  
  Then open the **settings panel** and choose **Text detection**, **OCR**, **Inpainting**, and **Translation** from the dropdowns. New modules appear automatically. Set **device** (CPU/CUDA) and any model-specific options there.

- **Headless (no GUI):**  
  `python launch.py --headless --exec_dirs "[DIR_1],[DIR_2]..."`  
  For continuous mode (prompt for new dirs after each batch until you type `exit`):  
  `python launch.py --headless_continuous --exec_dirs "DIR_1,DIR_2,..."`  
  Settings are read from `config/config.json`. Ensure the chosen detector, OCR, and inpainter are installed and configured in that config.

- **Logical DPI (font/rendering):**  
  Set in **Config panel → General**: **Logical DPI (restart to apply)** — 0 = system default; use 96 or 72 if font scaling is wrong. Persisted in config and applied on next launch. You can still override once via the command line: `python launch.py --ldpi 96`.

### Command-line arguments (launch.py)

| Argument | Description |
|----------|-------------|
| `--proj-dir PATH` | Open project directory on startup |
| `--headless` | Run without GUI |
| `--headless_continuous` | Like headless; after finishing `--exec_dirs` prompts for new dirs (comma-separated) until you enter `exit`. |
| `--exec_dirs "DIR1,DIR2,..."` | Translation queue: process folders in sequence (comma-separated). Same as Batch queue in GUI. |
| `--ldpi N` | Logical DPI override (e.g. 96, 72). Overrides config once. |
| `--config_path PATH` | Config file to use (default: `config/config.json`) |
| `--export-translation-txt` | Save translation to txt file once RUN completed |
| `--export-source-txt` | Save source text to txt file once RUN completed |
| `--update` | Update the repository before launching |
| `--reinstall-torch` | Reinstall torch even if already installed |
| `--frozen` | Skip requirement checks |
| `--debug` | Debug mode |
| `--nightly` | Enable AMD Nightly ROCm |
| `--qt-api pyqt6\|pyside6\|pyqt5\|pyside2` | Qt API (default: pyqt6; pyqt5 on Windows 7) |

**Example (headless batch):**  
`python launch.py --headless --exec_dirs "C:\manga\ch1,C:\manga\ch2" --config_path config/config.json`

### AMD GPU (ROCm / ZLUDA)

On **AMD** GPUs you can use either **ZLUDA** (translation layer, works with the project’s Python 3.10/3.11 and CUDA builds) or **native ROCm on Windows** (as of AMD driver 2026.1.1; requires Python 3.12 and official ROCm PyTorch wheels).

- **ZLUDA:** Install an [AMD HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) version that matches your driver, then [ZLUDA](https://github.com/lshqqytiger/ZLUDA/releases). Add the ZLUDA folder and `%HIP_PATH%bin` to PATH, and replace CUDA DLLs as described in the upstream README. Use **Config → device: Cuda** for detection/OCR (inpainting often stays on CPU). First run can take several minutes while PTX compiles.
- **Native ROCm (Windows):** Supported on AMD driver **2026.1.1+**. Requires **Python 3.12** and the ROCm PyTorch stack. Use `launch_win_amd_nightly.bat` or `python launch.py --nightly` after installing the appropriate ROCm wheels (see upstream for current URLs and GPU support). For full step-by-step instructions, HIP SDK vs ZLUDA version table, and driver notes, see the [original BallonsTranslator README](https://github.com/dmMaze/BallonsTranslator) and expand the **“启用 AMD ROCm 显卡加速方法”** (Enable AMD ROCm GPU acceleration) section.

---

## 3. Text detection – all modules and how to run

Select the detector from the **Text detection** dropdown in the settings panel. Pair detection-only modules with an OCR (see [INSTALL_EXTRA_DETECTORS.md](docs/INSTALL_EXTRA_DETECTORS.md) for which work with **none_ocr**).

**Original vs this fork:** **ctd**, **paddle_det**, **easyocr_det**, **ysgyolo**, **stariver_ocr** are from the original BallonsTranslator. **paddle_det_v5** (PP-OCRv5 detection) and all other detectors in the "New" table were added in this fork.

### Original / built-in

| Module | How to run | Notes |
|--------|-------------|--------|
| **ctd** | Select **ctd**; set device, detect_size, box score threshold, merge tolerance, etc. | ComicTextDetector; primary manga detector. See [Settings](#71-ctd-comictextdetector) below. |
| **paddle_det** | Select **paddle_det**; needs `paddlepaddle`, `paddleocr`. | Paddle OCR detection; pair with paddle_ocr or paddle_rec_v5. |
| **easyocr_det** | Select **easyocr_det**; needs `easyocr`. | CRAFT-based; pair with easyocr_ocr. |
| **ysgyolo** | Select **ysgyolo**; put YOLO `.pt` in `data/models/` with name starting with `ysgyolo` (e.g. `ysgyolo_comic_speech_bubble_v8m.pt`). | For comic bubble detection; pair with any OCR. |
| **stariver_ocr** | Select **stariver_ocr**; fill User/Password in params. | API returns boxes+text; set OCR to **none_ocr**. |

### New in this fork

| Module | Dependencies | How to run |
|--------|--------------|------------|
| **hf_object_det** | `pip install transformers torch` | Select **hf_object_det**. Default **model_id** is `ogkalu/comic-text-and-bubble-detector` (bubble, text_bubble, text_free). Change **model_id** for other HF object-detection models; use **score_threshold** and **labels_include** (comma-separated) to filter. |
| **rtdetr_comic** | `pip install transformers torch` | Select **rtdetr_comic**. Uses **RT-DETRv2** comic text & bubble detector; **no model id required** — leave **model_id** empty for default `ogkalu/comic-text-and-bubble-detector` (bubble, text_bubble, text_free). Set **score_threshold**, **box_padding**, optional **class_ids** (0,1,2) to filter. |
| **mmocr_det** | `pip install openmim` then `mim install mmengine mmcv mmdet mmocr` (see `doc/INSTALL_MMOCR.md` for Windows). | Select **mmocr_det**; pair with mmocr_ocr. |
| **paddle_det_v5** | `pip install paddlepaddle paddleocr` (3.x). | Select **paddle_det_v5**; pair with paddle_rec_v5. |
| **surya_det** | `pip install surya-ocr` | Select **surya_det**; pair with surya_ocr. |
| **magi_det** | `pip install transformers torch einops` | Select **magi_det**; model downloads from HF (ragavsachdeva/magi) on first use; pair with any OCR. |
| **craft_det** | `pip install craft-text-detector torch` | Select **craft_det**; outputs 4-point quads for merge; pair with any OCR. **Conflict:** needs opencv<4.5.4.62; see [Optional dependencies](#8-optional-dependency-conflicts-and-workarounds). |
| **rapidocr_det** | `pip install rapidocr-onnxruntime` | Select **rapidocr_det**; lightweight ONNX detection; pair with **rapidocr** OCR for EasyScanlate-like pipeline. Optional models in `data/models/rapidocr/`. See docs/OPTIONAL_DEPENDENCIES.md. |
| **dptext_detr** | Clone [DPText-DETR](https://github.com/ymy-k/DPText-DETR), install its deps. | Select **dptext_detr**; set **repo_path** to your clone; pair with any OCR. |
| **swintextspotter_v2** | Clone [SwinTextSpotterv2](https://github.com/mxin262/SwinTextSpotterv2), install its deps. | Select **swintextspotter_v2**; set **repo_path**; use **none_ocr** if demo outputs text. |
| **hunyuan_ocr_det** | Same as hunyuan_ocr (transformers, etc.). | Select **hunyuan_ocr_det**; set OCR to **none_ocr** to keep spotter text. |
| **textmamba_det** | None (stub) | Selecting it raises an error until official code is released; use mmocr_det or surya_det meanwhile. |

---

## 4. OCR – all modules and how to run

Select the OCR from the **OCR** dropdown. Install only the dependencies for the modules you use.

**Original vs this fork:** **paddle_ocr**, **manga_ocr**, **easyocr_ocr**, **mmocr_ocr**, **mit32px/mit48px**, **google_vision**, **bing_ocr**, etc. are from the original BallonsTranslator. **paddle_rec_v5** (PP-OCRv5 recognition), **paddle_vl**, **PaddleOCRVLManga** (PaddleOCR-VL manga/server), and all other OCRs in the "New" table were added in this fork.

### Original / built-in

| Module | How to run |
|--------|-------------|
| **paddle_ocr** | Paddle stack; select and set language, device. Pair with paddle_det. |
| **manga_ocr** | Select **manga_ocr**; model in `data/models/manga-ocr-base` (auto-download). |
| **easyocr_ocr**, **mmocr_ocr** | Pair with corresponding detector. |
| **mit32px**, **mit48px**, **mit48px_ctc** | From manga-image-translator; select as needed. |
| **google_vision**, **bing_ocr**, **one_ocr**, **windows_ocr**, **macos_ocr**, **llm_ocr**, **stariver_ocr**, **none_ocr** | Select and configure (API keys, etc.). **llm_ocr** with provider **OpenRouter** offers free vision models in the dropdown (e.g. `openrouter/free`, Gemma/Gemini/Qwen VL). **none_ocr** = no OCR (use with spotters). |

### New in this fork

| Module | Dependencies | How to run |
|--------|--------------|------------|
| **paddle_rec_v5** | `pip install paddlepaddle paddleocr` (3.x). | PP-OCRv5 recognition; pair with paddle_det_v5. Select and set language, device. |
| **rapidocr** | `pip install rapidocr-onnxruntime` | Select **rapidocr**; lightweight ONNX recognition; pair with **rapidocr_det** for EasyScanlate-like pipeline. Optional rec/dict in `data/models/rapidocr/`. See docs/OPTIONAL_DEPENDENCIES.md. |
| **PaddleOCRVLManga**, **paddle_vl** | PaddleOCR-VL (manga or local server). | VLM manga or server; select and configure. |
| **paddleocr_vl_hf** | `transformers` 5.x | Select **paddleocr_vl_hf**; use prompt "OCR:"; 109 languages, SOTA document. |
| **surya_ocr** | `pip install surya-ocr` | Select **surya_ocr**; set language (e.g. Chinese (Simplified)), **Fix Latin misread** True for CJK; crop_padding 6–8. |
| **trocr** | `transformers`, `torch`, `PIL` | Select **trocr**; good for printed/handwritten English. |
| **got_ocr2** | `transformers`, `torch` | Select **got_ocr2**; unified OCR, tables/formulas. |
| **glm_ocr** | `transformers` (e.g. 5.x) | Select **glm_ocr**; 0.9B document. |
| **donut** | `transformers`, `torch` | Select **donut**; DocVQA/CORD task prompts. |
| **qwen2vl_7b** | `transformers`, `torch`, `accelerate` | Select **qwen2vl_7b**; ~16GB+ VRAM. |
| **deepseek_ocr** | `transformers`, `trust_remote_code` | Select **deepseek_ocr**; document, layout. |
| **lighton_ocr** | `transformers`, `torch` | Select **lighton_ocr**; 1B, strong per-parameter. |
| **chandra_ocr** | `pip install chandra-ocr` | Select **chandra_ocr**; 9B, layout/tables. |
| **docowl2_ocr** | `transformers`, `trust_remote_code` | Select **docowl2_ocr**; document understanding. |
| **nanonets_ocr** | `transformers`, `torch` | Select **nanonets_ocr**; 3B VLM, chat-style. |
| **ocean_ocr** | `transformers`, `torch`, `einops` | Select **ocean_ocr**; 3B MLLM, quality-focused. |
| **internvl2_ocr**, **internvl3_ocr** | `transformers`, `torch` | Select and choose model size (2B/8B etc.); trust_remote_code. |
| **hunyuan_ocr** | `transformers`, `torch` (see HunyuanOCR repo) | Select **hunyuan_ocr**; SOTA <3B class. |
| **florence2_ocr** | `transformers`, `torch` | Select **florence2_ocr**; Microsoft vision, base/large. |
| **minicpm_ocr** | `transformers`, `torch` | Select **minicpm_ocr**; compact VLM. |
| **ocrflux** | `transformers`, `torch` | Select **ocrflux**; document OCR. |
| **manga_ocr_mobile** | `pip install tflite-runtime huggingface_hub transformers` (optional) | Select **manga_ocr_mobile**; TFLite Japanese manga; lighter than manga_ocr. |
| **nemotron_ocr** | `transformers`, `accelerate`, `torch`, `albumentations`, `timm`; postprocessing from HF repo. | Select **nemotron_ocr**; full-page document parsing; assigns text to blocks by bbox overlap; set **min_resolution** (e.g. 1024), **iou_threshold**. |

---

## 5. Inpainting – all modules and how to run

Select the inpainter from the **Inpainting** dropdown. Key settings: **inpaint_size** (max side before resize), **mask_dilation** (for lama_large_512px only). See [Settings reference](#7-settings-reference-inpaint-size-mask-dilation-detection-ocr-formatting) below.

### Original / built-in

| Module | How to run |
|--------|-------------|
| **aot** | Select **aot**; **inpaint_size** 1024 or 2048; device cuda/cpu. |
| **lama_mpe** | Select **lama_mpe**; **inpaint_size** 1024 or 2048; device cuda. |
| **lama_large_512px** | Select **lama_large_512px**; **inpaint_size** 512/768/1024/1536/2048; **mask_dilation** 0–5; device cuda; precision bf16/fp32. Best for manga text removal. |
| **patchmatch**, **opencv-tela** | Select for CPU/lightweight inpainting. |

### New in this fork

| Module | Dependencies | How to run |
|--------|--------------|------------|
| **simple_lama** | `pip install simple-lama` or `simple-lama-inpainting` | Select **simple_lama**. **Conflict:** pillow<10 required; see [Optional dependencies](#8-optional-dependency-conflicts-and-workarounds). |
| **lama_onnx** | `pip install onnxruntime`; download ONNX from Hugging Face opencv/inpainting_lama. | Select **lama_onnx**; set **model_path** to the `.onnx` file; **inpaint_size** 512/768/1024. |
| **lama_manga_onnx** | `pip install onnxruntime`; download mayocream/lama-manga-onnx. | Select **lama_manga_onnx**; set **model_path**; **inpaint_size** (e.g. 1024). |
| **diffusers_sd_inpaint** | `pip install diffusers transformers accelerate` | Select **diffusers_sd_inpaint**; **inpaint_size** (e.g. 512); device; prompt-based. |
| **diffusers_sd2_inpaint** | Same | Select **diffusers_sd2_inpaint**; 768 default. |
| **diffusers_sdxl_inpaint** | Same | Select **diffusers_sdxl_inpaint**; 1024; heavier. |
| **dreamshaper_inpaint** | Same | Select **dreamshaper_inpaint**; 512. |
| **flux_fill** | Same | Select **flux_fill**; enable **CPU offload** if VRAM limited. |
| **kandinsky_inpaint** | Same | Select **kandinsky_inpaint**. |
| **fluently_v4_inpaint** | Same | Select **fluently_v4_inpaint**; anime/comic style. |
| **cuhk_manga_inpaint** | Clone MangaInpainting repo, download checkpoints. | Select **cuhk_manga_inpaint**; set **repo_path** and **checkpoints_path**; line map auto-generated. |
| **repaint** | Same Diffusers stack | Select **repaint**; e.g. google/ddpm-ema-celebahq-256; 256×256. |
| **qwen_image_edit** | Same | Select **qwen_image_edit**; Qwen/Qwen-Image-Edit; heavy; **inpaint_size** (e.g. 1024). |
| **mat** | Clone [MAT](https://github.com/fenglinglwb/MAT), download checkpoint. | Select **mat**; set **repo_path** and **checkpoint_path** to `.pth`; **inpaint_size** 512. |

---

## 6. Translation modules

This fork extends the original translator list with both **classic MT** and **LLM-based** backends.

- **LLM / API-based translators**
  - **LLM_API_Translator** – unified OpenAI / Google / Grok / OpenRouter / local (LLM Studio) client with JSON-mode parsing, context + glossary support, and keyword substitutions. Recommended for GPT‑4o / Claude / Gemini / OpenRouter free models.
  - **Gemini\_neverliie** – Google **Gemini** via [`neverliie-ai-sdk`](https://pypi.org/project/neverliie-ai-sdk/). EasyScanlate-style usage: set an API key in **Config → Translator → Gemini\_neverliie → api_key**, optionally override **model** (default `gemini-1.5-flash`). Prompts are tuned for manga/manhwa dialogue; no special project format is required.
  - **Mistral\_neverliie** – **Mistral** via `neverliie-ai-sdk` (default model `mistral-small-latest`). Configure in the same place as Gemini\_neverliie with your Mistral key and (optionally) a custom model name.
  - **ChatGPT**, **trans\_chatgpt\_exp** – legacy OpenAI chat translators (kept for compatibility; new work should prefer **LLM_API_Translator** or neverliie-based modules).

- **Classic MT and specialized translators**
  - **google** – unofficial Google Translate HTML API (works out of the box without an API key; set your own key for higher limits).
  - **DeepL / DeepLx / DeepLx\_API** – DeepL official and community endpoints.
  - **nllb200**, **m2m100**, **t5\_mt**, **opus\_mt** – local Hugging Face models (Transformer-based MT).
  - **Sakura**, **Sugoi**, **EZTrans**, **Caiyun**, **Papago**, **Baidu**, **Youdao**, **Yandex** – JP↔EN and CN↔EN engines.
  - **trans\_ensemble** – ensemble (3+1) judge-style translator which can mix several engines (e.g. Google, nllb200, LLM\_API\_Translator, Gemini\_neverliie) and pick the best candidate per line.
  - **None**, **Copy Source** – utility translators for skipping MT or copying source text into the translation field.

Configure API keys and endpoints for each module in the settings panel (Config → DL Module → Translator). **LLM_API_Translator** with provider **OpenRouter** includes free text models in the model dropdown (e.g. `openrouter/free`, Llama, Gemma, StepFun, Qwen); see [OpenRouter free models](https://openrouter.ai/models?fmt=cards&max_price=0&order=most-popular&output_modalities=text&input_modalities=text). See original README and `doc/加别的翻译器.md` for adding additional custom translators.

### 6.1 Translation context and glossary

The **LLM_API_Translator** supports **cross-page and cross-chapter terminology consistency** so that long works (e.g. cultivation manhua) keep terms and style consistent across all images and chapters. This is implemented via:

- **Glossary (translator + project + series):** Term list (source → target) injected into the prompt so the model uses the same wording every time. You can set a translator-level glossary in Config → Translator params, and a **project glossary** in the Translation context dialog (Edit → Translation context (project)... or Translator config → Translation context (project)...). When a **series context path** is set, a series-level glossary is loaded from `data/translation_context/<series_id>/glossary.txt` and merged with the others.
- **Previous-page context:** When translating in reading order, the last N pages’ source and translation are appended to the prompt (compact format) so the model sees recent choices and repeats style/terms.
- **Series-level storage:** For a given series (e.g. `urban_immortal_cultivator`), the app stores recent page context in `data/translation_context/<series_id>/recent_context.json`. When you open another chapter of the same series, the translator can seed from that stored context so consistency carries across chapters.
- **Context limit and summarization:** To respect model context limits, the previous-context block is capped by **context_max_chars** (Translator param). When the block would exceed that cap, you can either **truncate** (keep the tail) or **summarize**: the app asks the model to summarize the context (preserving key terms and style) in one extra API call, then uses the summary instead of dropping content. See [§12.7](#127-translation-context-glossary-series-previous-pages-summarization) for full detail.

Design and research are documented in **docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md**.

---

## 7. Settings reference: inpaint size, mask dilation, detection, OCR, formatting

### 7.1 CTD (ComicTextDetector)

- **detect_size:** e.g. 1280 (higher = better quality, slower). Up to 2400 supported.
- **box score threshold:** 0.35–0.48 typical; lower = more boxes (e.g. 0.42–0.45 for manhua).
- **merge font size tolerance:** e.g. 3.0; higher = merge more lines into one bubble. **If boxes are too large**, try **lowering** this (e.g. 2.0 or 1.5) so fewer lines are merged per box.
- **mask dilate size:** 2 default; kernel for mask dilation at detection stage.
- **min box area:** 0 or 100–200 to drop tiny noise.
- **custom_onnx_path:** Optional path to alternate ONNX (e.g. mayocream comic-text-detector); leave empty for default CTD ONNX.
- **box_padding:** Pixels added around each box (default 5). **If boxes are too large**, set to **0** (or 1–2) to stop adding extra margin.
- **box_shrink_px:** Shrink each box inward by this many pixels (0 = off). Use **4–12** when CTD boxes are too large; applied before box_padding.

### 7.2 Inpaint size and tiling (all inpainters)

- **lama_large_512px:** Options 512, 768, 1024, 1536, 2048. Default 1024. Smaller = less VRAM, gentler on small bubbles; larger = more detail on big regions. Avoid 2048 unless needed (risk of artifacts).
- **Inpaint tiling (OOM fix):** Config → Inpainting (or DL Module) → **Inpaint tile size** (0 = off; 512–1024 for OOM), **Inpaint tile overlap** (64 px). Tiling activates when image is > 1.5× tile size; tiles are processed separately and stitched with overlap blending. Use only when you get out-of-memory errors; tiling can cause grey or blurry bubbles.
- **Exclude labels from inpainting (optional):** Config → Inpainting → **Exclude certain labels from inpainting** (off by default). When enabled, enter comma-separated detector labels (e.g. `other, scene`); blocks with those labels are not inpainted (e.g. leave scene text or signs as-is). Requires a detector that sets block labels (e.g. YSG YOLO). When off, all detected text regions are inpainted.
- **Inpaint full image (workaround for broken per-block):** Config → Inpainting → **Inpaint full image (no per-block crops)**. When enabled, the whole image is inpainted at once instead of cropping per text block. Uses more VRAM and can be slower, but avoids crop/mask issues that cause bad results with some models (e.g. all Lama variants). **Try this first if inpainting looks wrong.** Alternatively use **opencv-telea** or **patchmatch** (Config → Inpainting) — no model download, CPU-only, often more reliable on difficult pages.
- **Dark blobs or corruption even with full image:** If you still get a large dark smudge, white X, or glitchy area after enabling "Inpaint full image," the cause is usually (1) **Lama** struggling with that page, or (2) the **mask** covering too much (e.g. one big detected region). **Do this:** Switch to **opencv-telea** or **patchmatch** (Config → Inpainting) and run inpainting again on the same page — they use a different algorithm and often avoid the artifact. Also try: set **mask_dilation** to **0** (Inpainter params, for lama_large_512px), then **re-run detection** so the mask is rebuilt from text blocks only; then run inpainting again.
- **lama_mpe, aot:** 1024 or 2048.
- **Diffusers-based (SD, SD2, SDXL, DreamShaper, Fluently, Kandinsky, RePaint, Qwen-Image-Edit):** Each has an **inpaint_size** (or similar) in params; 512/768/1024 typical. Match to model native resolution when possible.
- **lama_onnx:** 512 (model is 512×512); param controls max side before resize.
- **lama_manga_onnx:** Often 1024 default; stride 64 for alignment.
- **mat:** 512 typical.

### 7.3 Mask dilation (kernel)

- **lama_large_512px** exposes **mask_dilation** (0–5). It sets **mask_dilation_iterations** for a 3×3 morphological dilation on the mask before inpainting. **0** = no dilation; **2** = default (balanced); **3–5** = more coverage for dots/smudges; **0–1** = minimal distortion on tiny bubbles.
- Base inpainter (`modules/inpaint/base.py`) applies this dilation in `inpaint()` so all block-based inpainters that inherit (e.g. lama_large_512px) use it. Other inpainters (AOT, lama_mpe, Diffusers, etc.) do not expose a separate mask_dilation param; the base applies a configurable **mask_dilation_iterations** only when the inpainter sets it (lama_large_512px).

### 7.4 Box padding (detectors) and crop padding (OCR)

- **Box padding (detectors):** CTD, Paddle, EasyOCR, YSGYolo, HF object-det, MMOCR, Surya expose **box_padding** (px). Uses `box_utils.expand_blocks()` to expand each box outward by padding on all sides before passing to OCR. **4–6 px** recommended to reduce clipped punctuation (?, !) and character edges. Clamped to image bounds.
- **Crop padding (OCR):** Many OCRs have **crop_padding** (pixels to add around each detected box when cropping for OCR). Typical range 0–24. **6–8** is a good default. Supported by: manga_ocr, easyocr, mmocr, manga_ocr_mobile, got_ocr2, ocrflux, internvl3, minicpm, florence2, ocean, hunyuan, paddleocr_vl_hf, qwen2vl, chandra, internvl2, nanonets, docowl2, deepseek, lighton, donut, glm_ocr, trocr, surya_ocr, and others.

### 7.5 Text and box formatting

- **Global font format:** In settings panel → **嵌字** (typesetting), the “global font format” is the format used when no text block is selected; you can set default font, size, color, alignment, etc.
- **Per-block formatting:** In text edit mode, select a block and use the right-hand font/format panel (bold, italic, underline, alignment, letter spacing, line spacing, vertical text). Supports rich text and presets.
- **Box/block layout:** Detection produces boxes (quadrilaterals); the app keeps **lines** (polygon points) per block. Merge/split and reading order depend on detector and post-processing (e.g. CTD merge tolerance, Paddle strict bubble mode). No changes to core box data format in this fork; new detectors return the same `(mask, List[TextBlock])` interface.

### 7.6 Paddle detection (strict bubble mode)

For **paddle_det**, **Strict bubble mode** applies stricter thresholds and filters (min_detection_area, max_aspect_ratio, box_shrink_px, merge_same_line_only, merge_line_overlap_ratio). Useful for comics so different bubbles are not merged. **det_limit_side_len** can be set (e.g. 960 when using Ocean OCR on CPU to avoid timeout).

### 7.7 Config panel (General, Save, DL Module)

Most behavior and display options are set in the **Config panel** (left bar → gear icon). All of the following are persisted in `config.json`.

- **General → Startup:** Reopen last project on startup; **Recent projects limit** (5–30); **Auto update from GitHub on startup** (check for updates and pull on launch — *can cause issues or bad results*, e.g. merge conflicts or broken code; use with caution; config and local files are not overwritten); **Logical DPI** (0 = system, 96/72 for font scaling; restart to apply); **Confirm before Run** (show Run/Continue/Cancel dialog); **After Run: Region merge** (Never / on all pages / on current page only — uses Region merge tool settings); **Dark mode** (synced with View → Dark Mode); **Display language** (UI language, same as View → Display Language); **Config panel font scale** (0.8–1.5, for this panel only).
- **General → OCR result:** **Spell check / Auto-correct OCR result** — after OCR, correct misspelled words when there is **exactly one suggestion** (e.g. "teh" → "the"); requires pyenchant and a system dictionary. Stored in `pcfg.ocr_spell_check`. Runs via `spell_check_textblocks` in OCR postprocess hooks (`utils/ocr_spellcheck.py`). See [OCR spell check / auto-correct](#ocr-spell-check--auto-correct-after-ocr) tutorial.
- **General → Typesetting:** Defaults for new/unchanged blocks: font size, stroke, color, alignment, writing mode, font family, effect (decide by program vs use global); **Auto layout**; **To uppercase**; **Independent text styles per project**; **Show only custom fonts**.
- **General → Save:** Result image format (PNG, JPG, WebP, JXL); **Quality**; **WebP lossless** (when format is WebP); Intermediate image format (PNG, JXL).
- **General → Saladict / search:** Show mini menu when selecting text; Saladict shortcut; Search engine URL for lookups.
- **DL Module:** **Default device** (used when a module’s device is "Default"); **Load model on demand**; **Empty run cache**; **Unload models after idle** (minutes, 0 = off). Detector/OCR/Inpainter/Translator dropdowns and their params (device, detect_size, crop_padding, etc.) are in the same panel. The **Translator** section includes **Translation context (project)...** and **Test translator**. *Note: Test translator may fail when Load model on demand or Unload models after idle is enabled — the translator is only loaded when you run a pipeline; run a page first or temporarily disable those options when testing.* See [§12.7](#127-translation-context-glossary-series-previous-pages-summarization) for translation context.

Module-specific params (CTD box score, mask dilation, inpaint_size, translator API keys, etc.) are in the corresponding Config sub-sections. See Config panel (§7) and §11–12 for implemented UI and workflow.

### 7.8 Full config reference (ProgramConfig & ModuleConfig)

| Config path | Key | Type | Default | Description |
|-------------|-----|------|---------|-------------|
| **General → Startup** | `open_recent_on_startup` | bool | true | Reopen last project on startup |
| | `auto_update_from_github` | bool | false | Check for updates and pull from GitHub on startup; can cause issues (see Config panel tooltip) |
| | `recent_proj_list_max` | int | 14 | Recent projects limit (5–30) |
| | `logical_dpi` | int | 0 | 0 = system; 96/72 for font scaling (restart to apply) |
| | `confirm_before_run` | bool | true | Show Run/Continue/Cancel dialog |
| | `darkmode` | bool | false | Dark mode (synced with View → Dark Mode) |
| | `display_lang` | str | — | UI language |
| | `config_panel_font_scale` | float | 1.0 | Config panel font scale (0.8–1.5) |
| **General → OCR** | Spell check / auto-correct | — | — | After OCR, correct misspellings (pyenchant) |
| **General → Typesetting** | `let_fntsize_flag`, `let_autolayout_flag`, etc. | — | — | Font size, auto layout, alignment, etc. |
| **General → Save** | `imgsave_ext` | str | '.png' | PNG, JPG, WebP, JXL |
| | `imgsave_quality` | int | 100 | Quality (1–100) |
| | `imgsave_webp_lossless` | bool | false | Lossless WebP (when format is WebP) |
| **DL Module** | `default_device` | str | '' | Device when module uses "Default" |
| | `load_model_on_demand` | bool | false | Load models only when needed |
| | `unload_after_idle_minutes` | int | 0 | Unload models after idle (0 = off) |
| **Module** | `enable_dual_detect` | bool | false | Run second detector |
| | `textdetector_secondary` | str | '' | Secondary detector name |
| | `inpaint_tile_size` | int | 0 | 0 = off; 512–1024 for OOM |
| | `inpaint_tile_overlap` | int | 64 | Overlap between tiles (px) |
| | `inpaint_exclude_labels_enabled` | bool | false | When true, exclude blocks by label (see **Exclude certain labels from inpainting**) |
| | `inpaint_exclude_labels` | str | '' | Comma-separated labels to exclude (e.g. `other, scene`); used when above is true |
| | `inpaint_full_image` | bool | false | When true, inpaint whole image at once (no per-block crops); try if Lama gives bad results. |
| **Manga source** | `manga_source_lang` | str | 'en' | Chapter feed language |
| | `manga_source_data_saver` | bool | false | Use data-saver images |
| | `manga_source_download_dir` | str | '' | Default: `~/BallonsTranslator/Downloaded Chapters` |
| | `manga_source_request_delay` | float | 0.3 | Seconds between API requests |
| | `manga_source_open_after_download` | bool | false | Open first chapter folder in app after download |
| **General** | `auto_region_merge_after_run` | str | 'never' | `never` \| `all_pages` \| `current_page` — auto Region merge after pipeline |
| | `region_merge_settings` | dict | — | Persisted Region merge tool settings |
| | `ocr_spell_check` | bool | false | Spell check / auto-correct OCR result (pyenchant) |
| | `shortcuts` | dict | — | Custom keybinds (action_id → key string) |

### 7.9 Keyboard shortcuts (customizable)

Open **View → Keyboard Shortcuts...** (Ctrl+K) to view and customize. Stored in `config.json`.

| Category | Action | Default key |
|----------|--------|-------------|
| **File** | Open Folder | Ctrl+O |
| | Save Project | Ctrl+S |
| **Edit** | Undo | Ctrl+Z |
| | Redo | Ctrl+Shift+Z |
| | Search (current page) | Ctrl+F |
| | Global Search | Ctrl+G |
| | Region merge tool | Ctrl+Shift+M |
| **View** | Drawing Board | P |
| | Text Editor | T |
| | Keyboard Shortcuts | Ctrl+K |
| | Context menu options | Ctrl+Shift+O |
| **Go** | Previous Page | PgUp |
| | Next Page | PgDown |
| | Previous (alt) | A |
| | Next (alt) | D |
| **Canvas** | Text block mode | W |
| | Zoom In | Ctrl++ |
| | Zoom Out | Ctrl+- |
| | Delete / Rect delete | Ctrl+D |
| | Inpaint (when drawing) | Space |
| | Select all blocks | Ctrl+A |
| | Escape / Deselect | Escape |
| | Delete (key) | Delete |
| | Create text box | Ctrl+Shift+N |
| **Format** | Bold | Ctrl+B |
| | Italic | Ctrl+I |
| | Underline | Ctrl+U |
| **Drawing** | Hand tool (pan) | H |
| | Inpaint brush | J |
| | Pen tool | B |
| | Rectangle select | R |

### 7.10 Drawing tools (DrawPanelConfig)

When in **Drawing Board** (P), the drawing panel exposes:

- **Pen tool (B):** `pentool_width`, `pentool_color`, `pentool_shape` — brush size and color.
- **Inpaint brush (J):** `inpainter_width`, `inpainter_shape`, `inpaint_hardness` (0–100; 100 = hard edge, 0 = soft/feathered).
- **Rectangle tool (R):** `rectool_auto`, `rectool_method`, `recttool_dilate_ksize`, `recttool_erode_ksize`.

Stored in `pcfg.drawpanel`. Changes apply immediately.

---

## 8. Optional dependency conflicts and workarounds

Some optional modules require dependency versions that **conflict** with the main `requirements.txt`. See **docs/OPTIONAL_DEPENDENCIES.md** for full detail.

| Module | Conflict | Workaround |
|--------|----------|------------|
| **craft_det** | `craft-text-detector` needs **opencv-python<4.5.4.62**; main app uses **opencv≥4.8**. | Use **easyocr_det** or **mmocr_det** instead; or install in a separate venv with older opencv. If you keep main opencv, **craft_det** may not register (version check in code) or fail at runtime. |
| **simple_lama** | `simple-lama-inpainting` needs **pillow<10**; project uses **Pillow 10.x**. | Use **lama_large_512px**, **lama_onnx**, or **lama_manga_onnx** instead; or downgrade Pillow in a separate venv. |

The main application and all other modules work with the versions in `requirements.txt`.

### Console / log messages (Windows)

- **`qt.qpa.screen: "Unable to open monitor interface to \\\\.\\DISPLAY1:" "Unknown error 0xe0000225."`**  
  If you see this in the console or logs, the monitor/display is likely **disabled in Windows Device Manager**. Re-enable the display adapter (e.g. **Device Manager → Display adapters → enable** or re-enable the monitor under **Monitors**). The app may still run, but Qt cannot open the display interface until a monitor is enabled.

---

## 9. New and modified files

### New files (no removals of original files)

- **Text detection:**  
  `modules/textdetector/detector_mmocr.py`, `detector_paddle_v5.py`, `detector_surya.py`, `detector_magi.py`, `detector_textmamba.py`, `detector_craft.py`, `detector_hf_object_detection.py`, `detector_rtdetr_v2.py`, `detector_dptext_detr.py`, `detector_swintextspotter_v2.py` (existing in original may differ; this fork adds or modifies as listed).

- **OCR:**  
  `modules/ocr/ocr_trocr.py`, `ocr_paddle_rec_v5.py`, `ocr_paddle_VL.py`, `ocr_paddleVL_manga.py`, `ocr_got_ocr2.py`, `ocr_glm_ocr.py`, `ocr_donut.py`, `ocr_paddleocr_vl_hf.py`, `ocr_qwen2vl.py`, `ocr_deepseek.py`, `ocr_lighton.py`, `ocr_chandra.py`, `ocr_docowl2.py`, `ocr_nanonets.py`, `ocr_ocean.py`, `ocr_internvl2.py`, `ocr_internvl3.py`, `ocr_florence2.py`, `ocr_minicpm.py`, `ocr_ocrflux.py`, `ocr_hunyuan.py`, `ocr_manga_mobile.py`, `ocr_nemotron.py`.

- **Inpainting:**  
  `modules/inpaint/inpaint_simple_lama.py`, `inpaint_diffusers_sd.py`, `inpaint_sd2.py`, `inpaint_sdxl.py`, `inpaint_dreamshaper.py`, `inpaint_flux_fill.py`, `inpaint_kandinsky.py`, `inpaint_fluently.py`, `inpaint_cuhk_manga.py`, `inpaint_repaint.py`, `inpaint_lama_onnx.py`, `inpaint_lama_manga_onnx.py`, `inpaint_qwen_image_edit.py`, `inpaint_mat.py`.

- **Documentation:**  
  `docs/BEST_MODELS_RESEARCH.md`, `docs/MODELS_REFERENCE.md`, `docs/QUALITY_RANKINGS.md`, `docs/OPTIONAL_DEPENDENCIES.md`, `docs/INSTALL_EXTRA_DETECTORS.md`, `docs/MANHUA_BEST_SETTINGS.md`, **`docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md`**, this README.  
  `doc/INSTALL_MMOCR.md` (if present).

### Unchanged (behavior and discovery)

- `modules/base.py`: `MODULE_SCRIPTS` and module discovery unchanged; new modules are picked up by existing `ocr_*.py`, `inpaint_*.py`, `detector_*.py` patterns.
- `launch.py`, config flow, and UI flow unchanged; new options appear in the same dropdowns.

### New or modified for UI / workflow / export (BallonsTranslatorPro)

- **Manga source:**  
  `utils/manga_sources/__init__.py`, `utils/manga_sources/mangadex.py` (MangaDex: search, feed, chapter-by-ID, at-home, download).  
  `utils/manga_sources/comick_source.py` (Comick Source API: search, get_feed; no download).  
  `utils/manga_sources/gomanga_api.py` (GOMANGA-API: search, feed, download via imageUrls).  
  `utils/manga_sources/manhwa_reader.py` (Manhwa-Reader Vercel API: /api/all, /api/info, /api/chapter; manhwa/webtoon).  
  `ui/manga_source_dialog.py` (dialog: source combo, search/URL/local folder UI, chapters list, download folder, config persistence; worker selects client by source_id).  
  `ui/export_dialog.py` (Export all pages dialog: **Also create PDF** checkbox, `get_also_pdf()`).  
  `ui/mainwindow.py` (`_do_batch_export`: optional img2pdf; `on_validate_project`: duplicate/overlapping block check via `union_area` from `utils/imgproc_utils`).

- **Keyboard shortcuts:**  
  `utils/shortcuts.py` (shortcut schema, default keybinds, `get_shortcut`, `get_shortcut_info`, `get_default_shortcuts`).  
  `ui/shortcuts_dialog.py` (ShortcutsDialog: filter, category, table with QKeySequenceEdit, reset row/all, apply/cancel; emits `shortcuts_changed`).  
  `ui/mainwindowbars.py` (LeftBar and TitleBar: `_shortcut_actions_*`, `apply_shortcuts`; View menu **Keyboard Shortcuts...**).  
  `ui/mainwindow.py` (all QShortcuts and drawing tool shortcuts created from `pcfg.shortcuts`; `apply_shortcuts`, `open_shortcuts_dialog`; View → Keyboard Shortcuts).

- **Canvas and context menu:**  
  `ui/canvas.py` — New signals: `merge_selected_blocks_signal`, `move_blocks_up_signal`, `move_blocks_down_signal`, `run_detect_region`, `copy_trans_signal`, `paste_trans_signal`, `clear_src_signal`, `clear_trans_signal`, `select_all_signal`, `spell_check_src_signal`, `spell_check_trans_signal`, `trim_whitespace_signal`, `to_uppercase_signal`, `to_lowercase_signal`. Context menu extended with Merge, Move up/down, Detect text in region/on page, Copy/Paste translation, Clear source/translation, Select all, **Spell check source text**, **Spell check translation**, **Trim whitespace**, **To uppercase**, **To lowercase**; **Merge** and **Move up/down** enabled only when selection state allows it; spell check, trim, and case actions enabled when at least one block is selected. **Text eraser tool:** When `image_edit_mode == TextEraserTool`, right-click does not start a stroke (avoids black lines); painting uses left-button; stroke item and hit-test in scene coordinates.

- **Main window and bars:**  
  `ui/mainwindowbars.py` — Tools menu: new action **Manga / Comic source...** (`manga_source_trigger`).  
  `ui/mainwindow.py` — Handlers: `on_merge_selected_blocks`, `on_move_blocks_up`, `on_move_blocks_down`, `on_run_detect_region`, `on_detect_region_finished`, `on_copy_trans`, `on_paste_trans`, `on_clear_src`, `on_clear_trans`, `on_select_all_canvas`, **`on_spell_check_src`**, **`on_spell_check_trans`**, **`on_trim_whitespace`**, **`on_to_uppercase`**, **`on_to_lowercase`**, `on_open_manga_source`; Save and Export all pages pass `webp_lossless` when WebP + lossless config; **`_do_batch_export`** supports optional PDF via img2pdf; **`on_validate_project`** includes overlapping-block check.

- **Scene text and module manager:**  
  `ui/scenetext_manager.py` — New method `swap_block_positions(i, j)` to swap two blocks in the project, scene, and text panel order.  
  `ui/module_manager.py` — New `run_detect_region(rect, img_array, page_name)` and signal `detect_region_finished(page_name, blk_list)` for region-based detection. **Dual text detection:** `_run_dual_detect(img, mask, blk_list, im_w, im_h)` instantiates the secondary detector from merged params (with internal keys stripped), runs secondary `detect()`, merges blocks by IoU threshold 0.4 and masks when same shape; called from pipeline when `enable_dual_detect` and `textdetector_secondary` are set and ≠ primary. Progress dialog detect bar label set to “Detecting (dual: primary + secondary):” when dual is active. Before each page translate: passes **series_context_path** into `set_translation_context`; after each successful translate calls **append_page_to_series_context** for series storage. Same for sequential and low-VRAM pipeline paths.

- **Translation context and glossary:**  
  `utils/series_context_store.py` — Series folder layout `data/translation_context/<series_id>/` with `glossary.txt` and `recent_context.json`; `get_series_context_dir`, `load_series_glossary`, `load_recent_context`, `append_page_to_series_context`, `merge_glossary_no_dupes`.  
  `utils/proj_imgtrans.py` — `translation_glossary`, **series_context_path**; save/load in `to_dict` / `load_from_dict`.  
  `modules/translators/base.py` — `set_translation_context(..., series_context_path=None)`, **append_page_to_series_context** (no-op in base).  
  `modules/translators/trans_llm_api.py` — Params: glossary, context_previous_pages, series_context_prompt, **series_context_path**, **context_max_chars**, **context_trim_mode**, **summarize_context_when_over_limit**. Loads series glossary and recent context; merges glossaries; builds previous-context block with trim and cap; when over cap and summarization on, **`_request_context_summary`** (one extra API call) to shorten context; else truncate. After each page, appends to series store.  
  `ui/translation_context_dialog.py` — Dialog to edit project **series context path** and **project glossary**; Save writes to `ProjImgTrans` and calls `save()`.  
  `ui/mainwindow.py` — `show_translation_context_dialog`; connects config panel and title bar **Translation context (project)...** to open dialog; requires project open.  
  `ui/mainwindowbars.py` — Edit menu action **Translation context (project)...** (`translation_context_trigger`).  
  `ui/module_parse_widgets.py` — Translator config: **Translation context (project)...** button, emits `show_translation_context_requested`. **Text detection panel:** **Run second detector (dual detect)** checkbox and **Secondary:** combobox; when dual is on and a secondary detector is selected, **Secondary detector params** block with lazy-loaded ParamWidgets per detector; `_on_secondary_param_edited` writes to `pcfg.module.textdetector_params[sec_name]`.

- **Config and config panel:**  
  `utils/config.py` — New `ProgramConfig` fields: `imgsave_webp_lossless`, `manga_source_lang`, `manga_source_data_saver`, `manga_source_download_dir`, `manga_source_request_delay`, `shortcuts`. **ModuleConfig:** **`enable_dual_detect`** (bool, False) and **`textdetector_secondary`** (str, '') for dual text detection. **`default_downloaded_chapters_dir()`** static method returns and creates `~/BallonsTranslator/Downloaded Chapters` when no download dir is set. Load config merges default shortcuts for any missing action.  
  `ui/configpanel.py` — WebP lossless checkbox in Save section, visibility tied to result format; load/save of `imgsave_webp_lossless` and manga_source_* (the latter only via the Manga source dialog). Sync of dual-detect checkbox and secondary detector combobox from `pcfg.module` in setupConfig.  
  `config/config.example.json` — Example keys for `imgsave_webp_lossless`, `manga_source_*`, and `shortcuts`.

- **I/O:**  
  `utils/io_utils.py` — `imwrite(..., webp_lossless=False)`; when WebP and `webp_lossless` True, uses quality 101 for lossless.

- **Documentation:**  
  `docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md` — Design and implementation of translation context, glossary, series storage, and LLM integration.

---

## 10. Fixes and behavior changes

- **lama_large_512px:**  
  - **check_need_inpaint = False** so the model always runs (no median fill skip); avoids “weird solid-color box” in speech bubbles.  
  - **mask_dilation** is configurable (0–5) via params; stored in **mask_dilation_iterations** and applied in base `inpaint()` with a 3×3 kernel.

- **lama_mpe:**  
  - Small-bubble normalization: when max side < 400, input is resized to at most 512 before inpainting to reduce over-strong inpainting and artifacts.

- **Base inpainter:**  
  - Mask dilation before inpainting uses **mask_dilation_iterations** (default 2) and a 3×3 kernel; only inpainters that set this (e.g. lama_large_512px) override the default.

- **hf_object_det:**  
  - Default **model_id** set to **ogkalu/comic-text-and-bubble-detector**; default **labels_include** = **bubble,text_bubble,text_free** so all three classes are used out of the box.

- **craft_det:**  
  - OpenCV version check: if opencv ≥ 4.5.4.62, **craft_det** is not registered and a warning points to **docs/OPTIONAL_DEPENDENCIES.md** and alternatives (easyocr_det, mmocr_det).

- **textmamba_det:**  
  - Implemented as a **stub**: registered and visible, but ** _load_model** and **_detect** raise a clear error until official code is released; message suggests mmocr_det or surya_det.

- **OCR crop_padding:**  
  - Multiple OCRs (e.g. manga_ocr, surya_ocr, paddle_rec_v5, trocr, manga_ocr_mobile) expose **crop_padding** (0–24) to add pixels around each box when cropping for OCR, reducing clipped text at edges.

- **Paddle det:**  
  - Strict bubble mode and params (det_limit_side_len, merge_same_line_only, merge_line_overlap_ratio, etc.) documented and used for comic workflows; see MANHUA_BEST_SETTINGS.md.

- **Dual text detection:**  
  - When **enable_dual_detect** is True and **textdetector_secondary** is set (and different from primary), the pipeline runs a second detector and merges blocks (IoU &lt; 0.4) and masks (when same shape). Secondary detector is built from merged config with internal keys (e.g. `__param_patched`) stripped so "Found invalid … config" warnings do not appear. Progress dialog shows **Detecting (dual: primary + secondary):** when dual is active. See [§12.9](#129-dual-text-detection-primary--secondary-detector).

- **Text eraser tool:**  
  - New drawing tool **Text eraser** (ImageEditMode.TextEraserTool) erases parts of text blocks by painting on the canvas; stroke is applied in scene coordinates and updates each block’s mask. Right-click does not start a stroke in this mode. Undo (Ctrl+Z) via **TextEraserUndoCommand**. See [§12.10](#1210-text-eraser-tool).

- **Config:**  
  - Defaults in `config/config.json` (e.g. ctd box score threshold, detect_size, inpainter choice) are unchanged unless you alter them; new modules appear when their dependencies are installed.  
  - **Config panel** (left bar → gear): General (Logical DPI, display language, dark mode, config font scale, recent projects, confirm before Run, OCR spell-check, typesetting defaults, save format/WebP lossless, Saladict/search), DL Module (default device, load on demand, unload after idle), and module-specific params. **Test translator** (Translator) runs a quick check of the current API. *Note: Test translator may fail if "Unload models after idle" or "Load model on demand" is on — the translator may not be loaded until you run a pipeline.* Typesetting options use a single-column layout with a minimum content width so controls do not go off screen. All persisted. See [§7.7](#77-config-panel-general-save-dl-module).  
  - **Proxy:** `utils/proxy_utils.py` provides `normalize_proxy_url`, `create_httpx_transport`, and `create_httpx_client`; the LLM API translator uses them so proxy strings without a scheme (e.g. `127.0.0.1:7897`) are normalized to `http://...` and work with httpx.  
  - UI/workflow/export: canvas context menu (detect in region/on page, merge, move up/down, copy/paste translation, clear, select all), **Translation context (project)** (Edit menu and Translator config dialog for series path and glossary), lossless WebP, **batch export to PDF** (Export all pages), **Check project** (missing files, invalid JSON, overlapping blocks), **Manga / Comic source** (Tools: MangaDex, GOMANGA, Manhwa Reader, Comick, Local folder), and keyboard shortcuts are documented in [§12](#12-ui-workflow-and-export-enhancements). **Quit:** Close confirmation ("Are you sure you want to quit?"). **Open menu:** Order is Open Folder → Open Images → Open Project. **Page list:** Context menu supports Reveal in File Explorer, Translate selected images, Remove from project. **Drag-drop:** Dropping image files on the canvas opens a project from that folder. Config fields `imgsave_webp_lossless`, `manga_source_*`, and `shortcuts` are persisted.

---

## 11. Documentation and references

| Document | Description |
|----------|-------------|
| **docs/TROUBLESHOOTING.md** | GPU OOM, HuggingFace gated models, provider API keys, dependency conflicts; quick reference table. |
| **docs/QUALITY_RANKINGS.md** | Tier-based quality/accuracy rankings for all detection, OCR, and translation modules; task-based SOTA (document vs manga vs multilingual); sanity-check notes. |
| **docs/MODELS_REFERENCE.md** | Map of recommended models to BallonsTranslator modules; quick reference and “not integrated” list. |
| **docs/BEST_MODELS_RESEARCH.md** | Detailed research on OCR, detection, inpainting; benchmarks and recommendations. |
| **docs/OPTIONAL_DEPENDENCIES.md** | craft_det and simple_lama dependency conflicts and workarounds; RapidOCR (rapidocr_det, rapidocr) install and optional models. |
| **docs/INSTALL_EXTRA_DETECTORS.md** | Optional detectors (SwinTextSpotter, DPText-DETR, CRAFT, hf_object_det); none_ocr usage; detection vs OCR coverage table. |
| **docs/MANHUA_BEST_SETTINGS.md** | Recommended detection, OCR, and inpainting settings for manhua (Chinese comics). |
| **docs/PROMPT_FIND_MANGA_DOWNLOAD_SOURCES.md** | In-depth prompt for ChatGPT/LLMs to find manga APIs with direct image URLs for implementing more download sources. |
| **docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md** | Design and implementation of translation context: glossary, previous-page context, series-level storage, and integration with LLM translator. |
| **doc/FORMATTING_COMPARISON_AI_VS_MAIN.md** | Comparison of formatting and layout behavior (webcomics/manhua) between this fork and BallonsTranslator-ai; useful when migrating or choosing settings. |
| **Original README** | [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) – base setup, Windows/Mac, translators, AMD ROCm/ZLUDA. |
| **doc/加别的翻译器.md** | How to add new translators. |

---

## 12. UI, workflow, and export enhancements

This section describes **all UI, workflow, and export-related changes** added in BallonsTranslatorPro: canvas context menu actions, block reordering and merging, lossless WebP export, **translation context (glossary, series, summarization)**, and the Manga / Comic source feature. These improvements make editing and exporting faster and support upstream feature requests (e.g. #1055 Lossless WebP, #1137 manual text detection).

### 12.1 Canvas right-click menu (text edit mode)

In **text edit mode**, right-clicking on the canvas (or after right-dragging a rectangle) opens a context menu with the following actions. **Menu items are enabled or disabled based on the current selection** so you only see actions that apply.

| Action | Description | When enabled |
|--------|-------------|--------------|
| **Copy** / **Paste** | Standard clipboard copy/paste (source text when blocks are selected). | Always. |
| **Copy translation** | Copy the translation text of all selected blocks to the clipboard (one block per line, newlines in a block become spaces). | When at least one block is selected. |
| **Paste translation** | Paste from clipboard into the translation fields of selected blocks; lines are split by newline and assigned to blocks in order (first line → first selected block, etc.). | When at least one block is selected. |
| **Copy source text** / **Paste source text** | Same as Copy/Paste but for source text (shortcuts Ctrl+Shift+C / Ctrl+Shift+V). | Always. |
| **Delete** | Delete selected blocks (Ctrl+D). | When at least one block is selected. |
| **Delete and Recover removed text** | Delete blocks but keep their text in a recover list (Ctrl+Shift+D). | When at least one block is selected. |
| **Clear source text** / **Clear translation** | Clear the source or translation text of all selected blocks. Marks the project as unsaved. | When at least one block is selected. |
| **Select all** | Select all text blocks on the current page (same as Ctrl+A in the text panel). | Always. |
| **Spell check source text** | Run spell check / auto-correct on the **source text** of selected blocks. Uses pyenchant; if a word is misspelled and there is exactly one suggestion, it is replaced (e.g. "teh" → "the"). Requires **pyenchant** and a system dictionary (e.g. en_US). If unavailable, a warning is shown. | When at least one block is selected. |
| **Spell check translation** | Same as above, applied to the **translation** text of selected blocks. | When at least one block is selected. |
| **Trim whitespace** | Remove leading and trailing whitespace from each line of source and translation in selected blocks. | When at least one block is selected. |
| **To uppercase** / **To lowercase** | Convert all source and translation text in selected blocks to uppercase or lowercase. | When at least one block is selected. |
| **Merge selected blocks** | Merge all selected blocks into a single block: source texts and translations are concatenated (with newlines), the first block (by index) is updated, and the other selected blocks are removed from the project and scene. | **Only when two or more blocks are selected.** |
| **Move block(s) up** | Move the selected block one position up in the block list (and in the right-hand text panel). Implemented by swapping the block with the one above. | **Only when exactly one block is selected and it is not the first block.** |
| **Move block(s) down** | Move the selected block one position down. | **Only when exactly one block is selected and it is not the last block.** |
| **Apply font formatting** / **Auto layout** / **Reset Angle** / **Squeeze** | Existing formatting and layout actions. | As before. |
| **Gradient type** / **Text on path** | **Gradient type** submenu: Linear or Radial. **Text on path** ([#1138](https://github.com/dmMaze/BallonsTranslator/issues/1138)) submenu: None, Circular, or Arc — draws text along a circle or arc (for balloons and SFX). Arc span is set in the format panel (Arc degrees). | When at least one block is selected. |
| **Detect text in region** | Run the current **text detector** only on the region you drew: right-drag a rectangle on the canvas, then right-click and choose this item. New text blocks are added in full-image coordinates and appended to the current page. No OCR or translation is run. Useful for adding bubbles in one area without re-running the full pipeline (addresses upstream #1137). | **Only when a rubber-band rectangle was drawn** (right-drag before right-click). |
| **Detect text on page** | Run the text detector on the **entire page** (same as running the pipeline with only detection enabled). New blocks are appended. Convenience alternative to opening the Run menu. | Always (when in text edit mode). |
| **Translate** / **OCR** / **OCR and translate** / **OCR, translate and inpaint** / **Inpaint** | Existing pipeline actions. | As before. |

**Implementation notes:**

- **Create text box:** If you right-drag a rectangle before opening the context menu, **Create text box** uses that rectangle’s size and position (mapped to base layer coordinates and clamped to the image bounds). Otherwise the box is created at the right-click position with the default size.
- **Merge:** The first selected block (by index) keeps its position; source and translation strings of all selected blocks are joined with newlines. Other selected blocks are removed from `imgtrans_proj.pages[page_name]` in **descending index order** (to avoid index shift), then removed from the scene and text panel via `SceneTextManager.deleteTextblkItemList`.
- **Move up/down:** Implemented in `SceneTextManager.swap_block_positions(i, j)`: the two blocks are swapped in the project page list, in `textblk_item_list`, and in `pairwidget_list`; `.idx` is updated on both blocks and their pair widgets; the two widgets are removed from and re-inserted into the text panel layout so the on-screen order matches the new block order.
- **Detect in region:** The canvas stores the last rubber-band rect. When you choose “Detect text in region”, that rect is passed to `ModuleManager.run_detect_region(rect, img_array, page_name)`, which crops the image, runs the detector, and offsets the resulting block coordinates back to full-image space. `detect_region_finished` is emitted; the main window appends the new blocks to the project and scene.

**Files:** `ui/canvas.py` (signals, menu build, enable/disable logic), `ui/mainwindow.py` (handlers: merge, move up/down, copy/paste translation, clear, select all, **spell check source/translation**, **trim whitespace**, **to uppercase/lowercase**, detect region), `ui/module_manager.py` (`run_detect_region`, `detect_region_finished`), `ui/scenetext_manager.py` (`swap_block_positions`). Spell check uses **utils/ocr_spellcheck.py** (`spell_check_line`); if pyenchant is not installed, the spell-check actions show a warning dialog.

---

### 12.2 Lossless WebP export (#1055)

When saving or exporting result images in **WebP** format, you can enable **lossless** encoding so the quality setting is ignored and no lossy compression is applied.

- **Config:** In **Config panel → General → Save**, when **Result image format** is set to **WebP**, a **“WebP lossless”** checkbox appears. It is stored in `ProgramConfig.imgsave_webp_lossless` and persisted in `config.json`.
- **Visibility:** The checkbox is only enabled when the format dropdown is WebP; for PNG/JPG/JXL it is disabled.
- **Save path:** When you **Save** the current page or use **Tools → Export all pages**, the save logic checks `pcfg.imgsave_ext == '.webp'` and `pcfg.imgsave_webp_lossless`; if both are true, it passes `webp_lossless=True` into the save call.
- **Backend:** `utils/io_utils.imwrite()` accepts a `webp_lossless` argument; when True and format is WebP, it sets `quality=101` so OpenCV uses lossless WebP encoding.

**Files:** `utils/config.py` (`imgsave_webp_lossless`), `ui/configpanel.py` (checkbox, `_update_webp_lossless_visibility`, load/save), `ui/mainwindow.py` (save and export all pages pass `webp_lossless`), `utils/io_utils.py` (`webp_lossless` and quality 101 for WebP).

---

### 12.3 Manga / Comic source (search, chapters, download)

A full **Manga / Comic source** workflow is available from **Tools → Manga / Comic source...**. The dialog supports **multiple sources**: **MangaDex** (search, chapter URL, download), **GOMANGA** (search, chapters, download), **Manhwa Reader** (manhwa/webtoon, search, chapters, download), **Comick** (search and chapter list only; no download), **Local folder** (open folder of images as project). It opens a dialog where you can search for manga/manhua/manhwa by title, list chapters by language, select chapters to download, and optionally open the downloaded folder in BallonsTranslator to translate.

#### Features

- **Two sources in one dialog (plus more):**
- **MangaDex:** Search by title → results list → select a title → **Load chapters** (with language filter) → chapter list with checkboxes → **Download selected chapters**.
- **MangaDex (raw / original language):** Choose **Raw language (chapters to load)** (e.g. Japanese, Korean, Chinese) → search by title (only manga with that original language) → Load chapters (chapters in that language) → download for translating in BallonsTranslator.
- **MangaDex (by chapter URL):** Paste a MangaDex chapter URL (or raw chapter UUID) → **Load chapter** → one chapter appears in the list → download as above.
  - **GOMANGA**, **Manhwa Reader:** Search → Load chapters → Download (full download support).
  - **Comick:** Search and chapter list only (no download; use MangaDex or open links in browser).
  - **Local folder:** Open a folder of images as a project (no search/download).
- **Page numbering for correct order:** Downloaded chapter images are saved as **001.ext**, **002.ext**, … (e.g. 001.png, 002.png) so that BallonsTranslator’s natural sort loads them in reading order. The original MangaDex filenames are not used for the saved files.
- **Language:** Choose translation language for the chapter feed (e.g. English, Japanese, Chinese Simplified/Traditional, Korean). Stored in config.
- **Quality:** **Use data-saver (smaller images)** uses MangaDex’s data-saver image set (smaller files, lower resolution); unchecked uses original quality. Stored in config.
- **Download folder:** Choose a base folder; each manga is saved in a subfolder named after the title, and each chapter in a subfolder named after the chapter (e.g. `Ch.1 – Vol.1`). **Default:** When no folder has been set, downloads use **`~/BallonsTranslator/Downloaded Chapters`** (e.g. `C:\Users\<You>\BallonsTranslator\Downloaded Chapters` on Windows); this folder is created automatically the first time the Manga source dialog is used. All preferences are persisted.
- **Request delay (rate limiting):** A **Request delay** spinbox (0–2 seconds) inserts a delay between each API request (search, feed, at-home, image fetches) to avoid overloading MangaDex. Default 0.3 s. Stored in config.
- **Open in BallonsTranslator:** After download, **Open folder in BallonsTranslator** opens the **first chapter folder** (so the project has images) via **Open project**, so you can run detection/OCR/translate immediately.

#### Config fields (persisted)

- `manga_source_lang` — Language code for chapter feed (e.g. `en`, `ja`, `zh-hans`).
- `manga_source_data_saver` — Use data-saver images when True.
- `manga_source_download_dir` — Last chosen download base directory; **if empty, default is `~/BallonsTranslator/Downloaded Chapters`** (created automatically).
- `manga_source_request_delay` — Seconds to wait between API requests (0–2).

These are saved when you change them in the dialog or when you close the dialog.

#### Implementation summary

- **MangaDex client** (`utils/manga_sources/mangadex.py`): `search(title, original_language=None)` (optional filter by original language for raw), `get_feed(manga_id, lang)`, `get_chapter_by_id(chapter_id)` for URL mode, `get_chapter_urls(chapter_id, data_saver)`, `download_chapter(...)` which saves pages as 001.ext, 002.ext, … . **Comick** (`comick_source.py`): search (NDJSON merge from all sources), get_feed; no download. **GOMANGA** (`gomanga_api.py`): search, get_feed, download_chapter (imageUrls); raises user-friendly error when upstream returns 403. **Manhwa Reader** (`manhwa_reader.py`): `is_available()` for source visibility; search via /api/all, get_feed via /api/info/:slug, download_chapter via /api/chapter/:slug.
- **Dialog** (`ui/manga_source_dialog.py`): Source combo (MangaDex, MangaDex raw/original language, MangaDex URL, Comick, GOMANGA, Manhwa Reader when available, Local folder); search/URL/local rows; results list; chapter list with checkboxes; language/quality/delay/folder; progress bar. Worker in QThread selects client by source_id; Manhwa Reader is shown only after availability check; Local folder: Open folder in BallonsTranslator triggers open_folder_requested(path).
- **Main window:** Tools menu **Manga / Comic source...**; connects open_folder_requested(path) to OpenProj(path) for chapter or local folder.

**Files:** `utils/manga_sources/__init__.py`, `mangadex.py`, `comick_source.py`, `gomanga_api.py`, `manhwa_reader.py`, `ui/manga_source_dialog.py`, `ui/mainwindowbars.py`, `ui/mainwindow.py`, `utils/config.py`, `config/config.example.json`. See §11–12 and Config panel. Export/Check: `ui/export_dialog.py` (Also create PDF), `ui/mainwindow.py` (`_do_batch_export` + img2pdf; `on_validate_project` + overlapping-block check).

---

### 12.4 Keyboard shortcuts (customizable keybinds)

A **Keyboard Shortcuts** dialog lets you view and customize keybinds for common actions. Shortcuts are stored in config and applied to the main window, title bar, left bar, and drawing panel.

- **Opening the dialog:** **View → Keyboard Shortcuts...** or default shortcut **Ctrl+K** (configurable in the same dialog).
- **Dialog features:** Filter by text or category; table of Category, Action, Shortcut (editable with **QKeySequenceEdit**); per-row **Reset** to default; **Reset all to default**; **Apply** (saves to config and updates all shortcuts immediately); **Cancel**.
- **Actions covered:** File (Open folder, Save project); Edit (Undo, Redo, Page search, Global search, Merge tool); View (Drawing Board, Text Editor, Keyboard Shortcuts, Context menu options); Go (Prev/Next page, alternate keys); Canvas (Textblock mode, Zoom in/out, Delete, Space, Select all, Escape, Delete line, **Create text box**); Format (Bold, Italic, Underline); Drawing tools (Hand, Inpaint, Pen, Rect). Keys are stored in portable form (e.g. `Ctrl+S`) and applied via `QKeySequence.fromString()`.
- **Config:** `ProgramConfig.shortcuts` is a dict `action_id → key string`. On load, any missing action is filled from defaults so all schema actions always have a binding. **Apply** in the dialog writes to `pcfg.shortcuts`, calls `save_config()`, and emits `shortcuts_changed`; the main window’s `apply_shortcuts()` updates all QShortcuts, QActions (title bar, left bar), and drawing panel tool tips.

**Files:** `utils/shortcuts.py` (schema, `get_default_shortcuts`, `get_shortcut_info`, `get_shortcut`), `utils/config.py` (`shortcuts` field, load_config shortcut merge), `ui/shortcuts_dialog.py` (ShortcutsDialog), `ui/mainwindowbars.py` (LeftBar and TitleBar `_shortcut_actions_*`, `apply_shortcuts`; View menu **Keyboard Shortcuts...**), `ui/mainwindow.py` (shortcut creation from config, `_shortcuts_list`, `_draw_shortcut_tools`, `apply_shortcuts`, `open_shortcuts_dialog`, connection to dialog’s `shortcuts_changed`).

---

### 12.5 Typesetting: auto-adjust text size to fit box, font size list, drag-and-drop (#1077)

These improvements address the original BallonsTranslator [feature request #1077](https://github.com/dmMaze/BallonsTranslator/issues/1077): automatic text-size adjustment based on text box dimensions, more font sizes in the list, and drag-and-drop support.

#### Auto-adjust text size to fit text box

You can have the program **automatically scale font size** so that text fits inside each balloon/text box while **keeping line structure unchanged** (no manual trial-and-error with font sizes). Use the **Text in box** dropdown in Config → General → Typesetting: **Auto fit to box** or **Fixed size (use font size list)** ([#1077](https://github.com/dmMaze/BallonsTranslator/issues/1077)).

- **How to enable:** In **Config panel → General → Typesetting**, set **Font Size** to **“decide by program”** and enable **Auto layout** (split translation into lines according to balloon region).
- **Behavior:** When both are on, the layout logic scales text up or down so it fits the detected region: it scales down if the laid-out text would overflow the bubble (with margin), and can scale up when the bubble is much larger than the text so short text stays readable. Line breaks and structure follow the source/region; only the font size is adjusted.
- **Implementation:** `ui/scenetext_manager.py` (auto layout and resize ratios), `utils/textblock.py` (font size normalization so text fits bubbles). Detection also sets `_detected_font_size` per block, which is used when “decide by program” is selected.

#### More font sizes in the list

The font size combo in the format panel (and when editing a text block) now includes **extra steps between 28 and 48** for finer control: in addition to 28, 36, 48, the list includes **30, 32, 34, 40, 44** (and the existing smaller/larger sizes). Scrolling or selecting from the list gives better precision without large jumps.

- **Where:** Format panel (right side) → Font size dropdown. Same list is used when a block has a fixed font size.
- **Implementation:** `ui/text_panel.py` → `FontSizeBox` → `SizeComboBox` item list.

#### Drag-and-drop to open project

You can **drag a folder or image files** onto the **canvas** to open them as a project (same as opening via File → Open folder/images). If drag-and-drop does not work on your system (e.g. some file managers or OS configurations), use **File → Open Folder...** or **Open Images...** (copy-paste workflow) instead; both paths are supported.

- **Implementation:** The main canvas has `setAcceptDrops(True)` and handles `dragEnterEvent` / `dropEvent`; when a folder or image list is dropped, it emits `drop_open_folder` or equivalent and the main window opens that path as the project. See `ui/canvas.py` (GraphicsView drop handling) and `ui/mainwindow.py` (connection to open project).

---

### 12.6 Batch processing queue (#1020)

[Issue #1020](https://github.com/dmMaze/BallonsTranslator/issues/1020) requested a batch processing queue for multiple files/folders with Pause/Cancel controls so users can queue many chapters and process them without manual intervention. This is available in **headless mode** via `--exec_dirs "dir1,dir2,..."`. The same behavior is now available in the GUI.

- **Opening the dialog:** **Tools → Batch queue...**
- **Queue list:** Add folders with **Add folder(s)...** (one folder per dialog) or **Add folder (include subfolders)** to add the selected folder and each of its **immediate subfolders** as separate queue items (e.g. one parent “Manga” folder → “Manga”, “Manga/Ch1”, “Manga/Ch2”, …). Remove selected items or **Clear all**.
- **Start queue:** Click **Start queue** to process the list in order. The app opens each folder as a project and runs the full pipeline (detect, OCR, translate, inpaint) as configured. When one folder is done, the next is opened automatically. The queue list in the dialog shrinks as each item is started.
- **Pause / Resume:** While the queue is running, **Pause** temporarily halts the pipeline (at page boundaries). **Resume** continues. Useful to free resources for other tasks.
- **Cancel queue:** **Cancel queue** stops the current job and clears the remaining queue (no further folders are processed).
- **Signals:** The main window emits `batch_queue_empty` when all items are done and `batch_queue_cancelled` when the user cancelled; the dialog updates its status text accordingly.

**Files:** `ui/batch_queue_dialog.py` (dialog UI), `ui/mainwindow.py` (run_batch, run_next_dir, signals, handlers), `ui/mainwindowbars.py` (Tools → Batch queue...), `ui/module_manager.py` (requestPause / requestResume in ImgtransThread and ModuleManager; pause wait in the page loop).

---

### 12.7 Other small behavior and UI details

- **Tools menu:** In addition to **Manga / Comic source...** and **Keyboard Shortcuts** (View), the **Tools** menu includes: **Manage models** (download/remove detection, OCR, inpainting models), **Batch export** (export all result images to a folder; option **Also create PDF** builds `exported.pdf` via img2pdf when checked), **Check project** (validate project: missing images, invalid JSON, **duplicate/overlapping text blocks** — reports page and block index pairs), **Re-run detection only** / **Re-run OCR only**, **Region merge tool**, and **Export all pages**. Display language and dark mode can also be toggled from **View**.
- **Config panel – WebP lossless:** The WebP lossless checkbox is shown in the Save section only when the result format is WebP; its **enabled** state is toggled by `_update_webp_lossless_visibility()` when the format combo changes, so you cannot turn on lossless when the format is not WebP.
- **Config panel – Typesetting and width:** The **Typesetting** section (Font Size, Stroke Size, Font Color, Stroke Color, Effect, Alignment, Writing-mode, Font Family) uses a single-column layout so all dropdowns stay visible; the config content area has a minimum width so **Test translator** (Translator) and other right-side controls do not go off screen. A horizontal scrollbar appears if the panel is narrow.
- **Test translator:** In **Config → DL Module**, the **Translator** section has a **Test translator** button that runs a short translation with the current API/key and shows success or error. **Note:** Having **Unload models after idle** or **Load model on demand** enabled can cause Test translator to fail sometimes (e.g. "No translator loaded"), because the translator may not be loaded until you run a pipeline. Run a page first, or temporarily disable those options when testing. **Proxy:** LLM API translator (and other modules using `utils/proxy_utils`) normalizes proxy URLs (e.g. `127.0.0.1:7897` → `http://127.0.0.1:7897`) so httpx accepts them without "Unknown scheme" errors.
- **Canvas – context menu state:** Before showing the context menu, the canvas computes the number of selected blocks and total blocks, then sets `setEnabled(True/False)` on the **Merge selected blocks**, **Move block(s) up**, **Move block(s) down**, **Spell check source text**, **Spell check translation**, **Trim whitespace**, **To uppercase**, and **To lowercase** actions so they appear grayed out when not applicable (e.g. merge when fewer than 2 selected, move up when none or first block selected; spell check, trim, and case when no blocks selected).
- **Export all pages:** Exported files are named **001.ext**, **002.ext**, **003.ext**, … in project page order (same convention as manga chapter downloads). Uses the same `imgsave_ext`, `imgsave_quality`, and `imgsave_webp_lossless` as the normal Save path. When **Also create PDF from exported images** is checked in the export dialog, the app builds `exported.pdf` in the chosen folder using `img2pdf` (user is prompted to install if missing).

---

### 12.8 Translation context: glossary, series, previous pages, summarization

This subsection documents **cross-page and cross-chapter translation context** for the LLM translator: how glossaries, previous-page context, series storage, and **context summarization** work together to keep terminology and style consistent. It applies only to **LLM_API_Translator**; other translators are unchanged.

#### Purpose

- **Terminology drift:** Without context, the same concept can be translated differently on each page (e.g. 丹田 → "dantian" on one page, "core" on another).
- **Style drift:** Tone and register can change from page to page.
- **Cross-chapter reuse:** For a long series (e.g. *Rebirth of the Urban Immortal Cultivator*), you want the same terms and style across all chapters. Storing context in a **series folder** lets the next chapter seed from the end of the previous one.

#### Glossary (three sources, merged)

1. **Translator param `translation_glossary`** — Multiline field in Config → Translator (LLM_API_Translator). Format: one entry per line, e.g. `source -> target` or `source = target`. Used on every page.
2. **Project glossary** — Stored in the project JSON and editable in the **Translation context (project)** dialog (Edit menu or Translator config panel). Same format; merged with the translator glossary at run time. Travels with the project.
3. **Series glossary** — When a **series context path** is set (project or translator param), the app loads `data/translation_context/<series_id>/glossary.txt`. Format: one line per entry, `source -> target` or `source = target`. Lines starting with `#` are ignored. Merged with translator and project glossaries; first occurrence of each source term wins.

The merged glossary is injected into the **system prompt** (or user prompt) so the model is instructed to use those exact translations when the source terms appear.

#### Series context path and folder layout

- **Series context path** can be:
  - A **series ID** (e.g. `urban_immortal_cultivator`): resolved to `data/translation_context/urban_immortal_cultivator/`.
  - A **path** (relative to program root or absolute): used as-is (relative paths are joined with the program root).
  - **Default:** If neither the project nor the translator sets a path, the app uses **`default`** (i.e. `data/translation_context/default/`) so context is never blank.
- **Folder layout** for a series:
  - `glossary.txt` — Optional. One line per entry: `source -> target`. Loaded when building the prompt.
  - `recent_context.json` — List of page entries `{"sources": [...], "translations": [...]}`. Last N pages are loaded to seed “previous context”; after each translated page, the pipeline appends that page to this file (and trims to a max, e.g. 15 pages).

So all chapters that share the same series ID/path share the same glossary and a rolling window of recent page context.

#### Previous-page context

- **When:** Only when translation runs in **reading order** (sequential pipeline or ordered queue). The pipeline passes the last **N** pages (param **context_previous_pages**, 0–5) that have already been translated.
- **What is passed:** For each of those pages, source texts and translations are formatted into a compact block (e.g. at most 2 lines per page in **compact** trim mode, or more in **full** mode). The block is labeled “Previous context (for terminology and style consistency):” and appended to the user prompt.
- **Seeding from series:** If a series path is set and there are few in-memory “previous” pages (e.g. start of a new chapter), the translator loads **recent_context.json** from the series folder and uses those entries as the initial previous pages, so the new chapter continues from where the last chapter left off.

#### Context limit: truncate vs summarize

- **context_max_chars** (Translator param, default 2000, 0 = no limit): The previous-context block (after trim) must not exceed this length so the full prompt fits the model’s context window.
- **When over limit:**
  - **Truncate (default when summarization is off):** Replace the block with `"...\n" + last max_chars characters`. Simple and no extra API call, but you lose the beginning of the context.
  - **Summarize (optional):** If **summarize_context_when_over_limit** (Translator checkbox, default True) is enabled, the app makes **one extra API call** before the translation request: it sends the long context to the model with a system prompt asking to shorten it to under `context_max_chars` while **preserving key term translations** (e.g. `[source] -> [translation]`), character names, and tone. The model’s reply is used as the previous-context block. If the reply is still too long, it is trimmed. If the summary call fails (e.g. no API key, network error), the code falls back to truncation. This way you retain more useful signal (terms and style) instead of discarding the start of the context.

**Params summary (LLM_API_Translator):**

| Parameter | Description |
|-----------|-------------|
| **translation_glossary** | Multiline: `source -> target` per line. Merged with project and series glossary. |
| **series_context_path** | Folder or series ID for cross-chapter consistency; uses `data/translation_context/<id>/` when ID. |
| **series_context_prompt** | Optional short prompt, e.g. “This is a cultivation manhua. Keep terms consistent.” |
| **context_previous_pages** | Number of previous pages (0–5) to include as context. 0 = off. |
| **context_max_chars** | Max characters for the previous-context block. 0 = no limit. |
| **context_trim_mode** | **full** = more lines per page; **compact** = at most 2 lines per page to save tokens. |
| **summarize_context_when_over_limit** | When context exceeds context_max_chars, ask the model to summarize it (one extra call) instead of truncating. |

#### UI: Translation context (project) dialog

- **Where to open:** **Edit → Translation context (project)...** or **Config → DL Module → Translator** section → **Translation context (project)...** button.
- **Requires:** A project must be open (File → Open Folder / Open Project). If none is open, a message asks you to open one first.
- **Contents:**
  - **Series context path** — Line edit. Folder or series ID (e.g. `urban_immortal_cultivator`). Leave empty to use only the translator’s **series_context_path** param (or no series).
  - **Project glossary** — Multiline text: one entry per line, `source -> target` (or `=`, `:`). Lines starting with `#` are ignored. Saved into the project JSON and merged at translate time.
- **Save:** On **Save**, the dialog writes `series_context_path` and `translation_glossary` to the current `ProjImgTrans` and calls `save()` so the project file is updated.

#### Pipeline integration

- **Before translating a page:** The pipeline (e.g. `ui/module_manager.py`) determines the **series_context_path** from the project’s `series_context_path` if set, else from the translator’s **series_context_path** param. It calls `translator.set_translation_context(previous_pages_data, glossary_override=None, series_context_path=series_path, ...)` so the translator can load the series glossary and recent context and set the previous-pages list.
- **After a successful translation:** If a series path is set, the pipeline calls `translator.append_page_to_series_context(series_path, sources, translations)`. The LLM translator appends that page to `recent_context.json` in the series folder (and trims to the configured max).

**Files:** `utils/series_context_store.py` (folder layout, load/save glossary and recent context, merge glossaries), `utils/proj_imgtrans.py` (`translation_glossary`, `series_context_path`, save/load), `modules/translators/base.py` (`set_translation_context`, `append_page_to_series_context`), `modules/translators/trans_llm_api.py` (params, prompt assembly, truncate/summarize, `_request_context_summary`), `ui/module_manager.py` (set context before translate, append after translate), `ui/translation_context_dialog.py` (dialog), `ui/mainwindow.py` and `ui/mainwindowbars.py` (Edit menu and config panel button), `ui/module_parse_widgets.py` (Translator **Translation context (project)...** button). Design doc: **docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md**.

### 12.9 Dual text detection (primary + secondary detector)

You can run **two text detectors** on each page and merge their results so that regions missed by one detector (e.g. captions vs speech bubbles) are caught by the other. This is useful when one detector is strong on bubbles and another on captions or SFX.

#### Config and UI

- **Config (ModuleConfig):** `enable_dual_detect` (bool, default False) and `textdetector_secondary` (str, default ''). Stored in `config.json` under module config.
- **Config panel → Text detection:**  
  - **Run second detector (dual detect)** — Checkbox; when checked, a second detector runs after the primary on every page.  
  - **Secondary:** — Dropdown listing all valid text detectors (same list as primary); select the second detector (cannot be the same as primary).  
  - **Secondary detector params:** — When dual detect is enabled and a secondary detector is selected, a **Secondary detector params** block appears below with the same param widgets (device, thresholds, etc.) as the primary panel. Edits are saved to `textdetector_params[secondary_name]` so the secondary detector uses them when instantiated during the pipeline.
- **Config load/save:** `configpanel.setupConfig` syncs the dual-detect checkbox and secondary combobox from `pcfg.module`; the panel’s `addModulesParamWidgets` is called with merged text detector params so both primary and secondary param sets are available.

#### Pipeline behavior

- **When:** After the primary detector runs (`mask, blk_list = self.textdetector.detect(...)`), if `enable_dual_detect` is True, `textdetector_secondary` is non-empty, and secondary ≠ primary, the pipeline calls `_run_dual_detect(img, mask, blk_list, im_w, im_h)`.
- **Secondary detector creation:** The secondary detector is instantiated from `TEXTDETECTORS.module_dict[sec_name]` with params from `merge_config_module_params(cfg_module.get_params('textdetector'), GET_VALID_TEXTDETECTORS(), TEXTDETECTORS.get)`. Internal keys (e.g. `__param_patched`) are stripped before passing params to the constructor to avoid “invalid config” warnings. If creation or model load fails, a warning is logged and the primary result is returned unchanged.
- **Merge logic:**  
  - Secondary’s `detect(img, proj)` is run. If it raises or returns no blocks, the primary result is returned (masks are still merged if the secondary returned a valid mask).  
  - For each secondary block, IoU (intersection-over-union) with every primary block is computed; if **max IoU < 0.4**, the secondary block is appended (low overlap = new region).  
  - Blocks are reordered with `sort_regions(blk_list)`.  
  - Masks are merged only when both primary and secondary masks exist and have the **same shape** (`np.bitwise_or(mask, mask2)`); if the secondary returns a different mask shape, only blocks are merged.  
- **Progress dialog:** When you start the pipeline (Run) or open the run-status window, the detection bar label is set to **Detecting (dual: &lt;primary&gt; + &lt;secondary&gt;):** when dual detect is enabled and a secondary is selected; otherwise **Detecting:**. This makes it clear that dual mode is active.

#### Files and robustness

- **Files:** `utils/config.py` (`enable_dual_detect`, `textdetector_secondary`), `ui/module_parse_widgets.py` (TextDetectConfigPanel: dual checkbox, secondary combobox, **Secondary detector params** container and lazy ParamWidgets, `_on_secondary_param_edited` to persist to `textdetector_params`), `ui/configpanel.py` (setupConfig sync), `ui/module_manager.py` (`_run_dual_detect`, pipeline call, progress label, param stripping).  
- **Robustness:** `blk_list` or `blk_list_2` may be None (treated as []). Secondary params come from merged config so defaults apply if a detector was never selected. Mask merge is skipped when shapes differ. Failures in secondary creation or detection are caught and logged; the run continues with the primary result.

---

### 12.10 Text eraser tool

A **text eraser** drawing tool lets you erase parts of text blocks by painting over them. The tool modifies each block’s **mask** (the region where text is rendered) so that painted areas are no longer drawn. This is useful for removing stray strokes, cleaning bubble edges, or hiding parts of a line without deleting the block.

#### How to use

- **Enable:** In the **drawing tools** bar (e.g. when in image-edit / draw mode), select the **Text eraser** tool (eraser icon). The canvas switches to `ImageEditMode.TextEraserTool`.
- **Erase:** Use the **left mouse button** to paint over the text you want to erase. The stroke is applied in **scene coordinates** so hit-testing matches the visible text blocks. If no text blocks are selected, the eraser applies to **all text items** whose `sceneBoundingRect()` intersects the stroke; if blocks are selected, only those blocks’ masks are modified.
- **Undo:** **Ctrl+Z** undoes the last eraser stroke (and other drawing commands). Each eraser stroke is recorded as a **TextEraserUndoCommand** storing the item and the mask state before/after.

#### Implementation details

- **Coordinates:** Stroke and mask updates use **scene space**: the stroke’s scene rect is computed with `stroke_item.mapToScene(QRectF(...))`; for each mask pixel the corresponding scene point is mapped into the text item’s space with `item.mapToScene` / `stroke_item.mapFromScene` so the correct mask pixels are cleared. The mask region is aligned with the block’s `_text_mask_region()` and the item’s bounding rect.
- **Events:** In **TextBlkItem**, when `scene.image_edit_mode == ImageEditMode.TextEraserTool`, **left-button** press/move/release are ignored by the item so the scene receives them and the stroke is drawn. **Right-click** does not start a stroke in text-eraser mode (to avoid accidental black lines or rubber-band while erasing); right-drag is used for selection/rubber-band only when not painting.
- **Merge:** After a stroke, `_apply_text_eraser_stroke` builds the combined stroke mask, intersects with each target block’s mask, and updates the block’s mask. The stroke item is removed from the canvas and the undo command is pushed.

**Files:** `ui/image_edit.py` (`ImageEditMode.TextEraserTool`), `ui/drawingpanel.py` (text eraser tool button, `on_finish_painting`, `_apply_text_eraser_stroke`), `ui/canvas.py` (mouse press: no stroke on right-click in TextEraserTool; addStrokeImageItem; painting vs rubber-band), `ui/textitem.py` (ignore left-button in TextEraserTool), `ui/drawing_commands.py` (`TextEraserUndoCommand`), `config/stylesheet.css` (DrawTextEraserTool indicator styles).

---

### 12.11 Text edit panel and right panel layout

The **right-hand panel** (text edit / format panel) and the **comic translation stack** have been adjusted so the default width is closer to the minimum and the layout is less cramped when many controls are shown.

- **Right panel width:** The right comic-trans stack panel has a **minimum width** (e.g. 320) and **maximum width** (e.g. 900). On show, the splitter is set so the right panel’s default size is **just above the minimum** (e.g. 322) so more space stays on the canvas by default.
- **Format / alignment controls:** The format bar (color, alignment, bold/italic/underline, letter spacing, line spacing, vertical text, text-on-path, arc degrees, warp, opacity) uses increased spacing and margins (`FONTFORMAT_SPACING` and related constants). The **opacity** label is given a minimum width so “Opacity” is not truncated (e.g. “Opacit”). Rows are split (e.g. `hl2` for color/align/format/vertical, `hl2b` for text-on-path, arc, warp, opacity) so that narrow windows do not squeeze every control onto one line.
- **Stylesheet:** In `config/stylesheet.css`, the **AlignmentChecker** and **QFontChecker** indicators (bold/italic/underline state) are made **smaller** so they fit better next to the font dropdown and other format controls.

**Files:** `ui/mainwindow.py` (splitter sizes, right panel min/max, showEvent default), `ui/text_panel.py` (layout rows, spacing, opacity label `setMinimumWidth`), `config/stylesheet.css` (AlignmentChecker, QFontChecker indicator size).

---

If a module is missing from the dropdown or fails to load, check the console/log for import errors and install the dependencies listed in the tables above. For quality-focused choices, use **docs/QUALITY_RANKINGS.md** and **docs/MANHUA_BEST_SETTINGS.md**.
