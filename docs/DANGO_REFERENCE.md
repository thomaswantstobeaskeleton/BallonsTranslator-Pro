# Dango-Translator (团子翻译器) — reference and implementation ideas

This document summarizes [PantsuDango/Dango-Translator](https://github.com/PantsuDango/Dango-Translator) and what could be implemented or reused in BallonsTranslator. The **current Dango app is v6.x and written in Golang**; the GitHub repo still contains the **legacy Python code (v4.5.8)**, which is no longer updated but is useful for understanding APIs and flow.

---

## 1. What Dango-Translator is and how it works

- **Product:** 团子翻译器 — “raw meat” translator: OCR + translation for games, manga, and screen content.
- **Modes:**
  - **Real-time translation:** Capture a screen region → OCR (online/offline/Baidu) → translate → show result. No manga-page pipeline.
  - **Image/manga translation (图片翻译):** Load image → detect + OCR (e.g. 星河云 manga API) → translate → erase text (inpaint) → render translated text (typeset). This is the part that overlaps with BallonsTranslator.
- **Account system:** Login (and in the old Python app, **register**) is in-app. Accounts are shared with 星河云 (Starriver); same credentials used for cloud OCR/quota. The **dashboard** ([dashboard.stariver.org.cn](https://dashboard.stariver.org.cn/)) is for login and purchasing; **registration** in current versions is done inside the **Dango Translator** app (download from [translator.dango.cloud](https://translator.dango.cloud/)).

### 1.1 Python repo layout (legacy, v4.5.8)

| Area | Role |
|------|------|
| `app.py` | Entry; login → translation UI; manga UI; register/forget-password from login window. |
| `ui/login.py`, `ui/register.py` | Login and **in-app registration** (no web register on dashboard). |
| `ui/manga.py` | Manga/image translation UI (detect, OCR, translate, inpaint, render). |
| `utils/http.py` | `loginDangoOCR`, `loginCheck`, `onlineOCRQueryQuota`, `mangaOCRQueryQuota`. |
| `translator/ocr/dango.py` | **Starriver/星河云 APIs:** login, online OCR, **mangaOCR**, **mangaIPT** (inpaint), **mangaRDR** (render), **mangaFontList**, **dangoTrans** (translate). |
| `utils/sqlite.py`, `ui/trans_history.py` | Translation history (DB + UI). |
| `ui/filter.py` | Filter/censor words (屏蔽词). |

---

## 2. Starriver / 星河云 APIs (shared with BallonsTranslator)

BallonsTranslator already uses the **same manga OCR API** as Dango for detection and (optionally) OCR.

| API | URL (from Dango) | BallonsTranslator |
|-----|------------------|--------------------|
| Login | `https://capiv1.ap-sh.starivercs.cn/OCR/Admin/Login` | ✅ Same (`detector_stariver.py`, `ocr_stariver.py`) |
| Manga OCR (detect + OCR) | `https://dl.ap-qz.starivercs.cn/v2/manga_trans/advanced/manga_ocr` or `.../ap-sh.../manga_ocr` | ✅ Same (detector: boxes+mask+text; OCR: per-crop or use detector text with `none_ocr`) |
| Text inpainting (消字) | `https://dl.ap-sh.starivercs.cn/v2/manga_trans/advanced/text_inpaint` | ❌ Not used (we use local inpainter, e.g. lama_large_512px) |
| Text render (嵌字) | `https://dl.ap-sh.starivercs.cn/v2/manga_trans/advanced/text_render` | ❌ Not used (we typeset locally) |
| Font list | `https://dl.ap-sh.starivercs.cn/v2/manga_trans/advanced/get_available_fonts` | ❌ Not used |
| Translate (私人团子) | `https://dl.ap-sh.starivercs.cn/v2/translate/sync_task` | ❌ Not used |

So today we only use **Login + manga_ocr**. The rest are optional extensions.

---

## 3. What could be implemented

### 3.1 Optional Starriver cloud inpainter

- **What:** Call `text_inpaint` with token + image + mask (base64) instead of local Lama/PatchMatch.
- **Why:** Offload GPU; same quality as Dango’s cloud erase; useful if user has no VRAM for inpainting.
- **Effort:** S–M. Add an inpainter backend that POSTs to the Starriver `text_inpaint` URL (reuse token from existing Starriver detector/OCR config).
- **Risk:** Low (optional; depends on Starriver quota/availability).

### 3.2 Optional Starriver cloud typesetting (text_render)

- **What:** Send inpainted image + translated text + text blocks (+ optional font) to `text_render`, get back a rendered image.
- **Why:** One-click “same as Dango” result; no local font/layout logic.
- **Effort:** M. Need to map our `TextBlock`/layout format to the API’s `text_block` and `translated_text`; handle `font_selector` if we add font list.
- **Risk:** Low. Optional; only for users who want cloud render.

### 3.3 Starriver “私人团子” as a translator

- **What:** New translator module that calls `dangoTrans` (sync_task): token + `texts` (list of lines) + `from` / `to` (e.g. `"CHS"`).
- **Why:** Use Dango’s own translation backend with the same account.
- **Effort:** S. One new translator class; reuse token from Starriver config.
- **Risk:** Low. Depends on Starriver translate quota/policy.

### 3.4 Starriver font list for typesetting

- **What:** Call `get_available_fonts` with token; show list in Config or in typesetting font dropdown.
- **Why:** If we add cloud render (3.2), or if we want to restrict local font choice to Starriver’s set.
- **Effort:** S.
- **Risk:** Low.

### 3.5 Dango v6 “local manga API” client (if documented)

- **What:** Dango v6.1.5 added “本地调用漫画API” (local manga API). If the Golang app exposes an HTTP API (e.g. on localhost), we could add a detector or pipeline step that sends an image and receives blocks/mask/text.
- **Why:** Reuse Dango’s full manga pipeline (detect + OCR + maybe translate) when Dango is running, without Starriver account.
- **Effort:** M (depends on API spec). Need to find docs or reverse-engineer the local API (e.g. in [Dango docs](https://dango-docs-v6.ap-sh.starivercs.cn/)).
- **Risk:** Low if optional; API may change with Dango versions.

### 3.6 Translation history (DB + UI)

- **What:** Persist source → translation per block or per page (e.g. SQLite); UI to search/reuse/export.
- **Why:** Consistency across pages; glossary-like reuse; matches Dango’s “翻译历史”.
- **Effort:** M–L. DB schema, save on translate, optional UI panel.
- **Risk:** Low. BallonsTranslator already has translation context/glossary; this would extend it with history.

### 3.7 Filter / censor words (屏蔽词)

- **What:** Configurable list of words/phrases to replace or hide in translation output (e.g. replace with `***` or remove).
- **Why:** Same idea as Dango’s filter; user-controlled censorship or term replacement.
- **Effort:** S. Apply post-processing on translator output using a small word list from config.
- **Risk:** Low.

### 3.8 Quota / validity display for Starriver

- **What:** Call Starriver’s quota/validity API (Dango uses `ocr_query_quota` with token; response has `Result` with `PackName`, `EndTime`) and show “Manga OCR valid until …” or “Out of quota” in Config.
- **Why:** User sees when the Starriver account expires or runs out.
- **Effort:** S. Need the exact quota URL (from Dango `dict_info["ocr_query_quota"]` or docs).
- **Risk:** Low.

---

## 4. What we do not need to port

- **Real-time screen capture + OCR + translate:** Different product focus (screen vs. manga pages). Not in scope for BallonsTranslator.
- **Full Dango UI (ranges, multiple translators, etc.):** We already have our own detector/OCR/translator/inpainter pipeline and config.
- **Offline OCR (local exe on port 6666):** Dango uses a separate local OCR process; we use in-process detectors/OCRs.
- **Registration flow:** Account creation stays in Dango app or Starriver; we only consume token (User/Password) as we do now. See [STARRIVER.md](STARRIVER.md).

---

## 5. Summary table

| Idea | Effort | Dango reference | Notes |
|------|--------|------------------|--------|
| Starriver cloud inpainter | S–M | `translator/ocr/dango.py` → `mangaIPT` | Optional backend; token + image + mask. |
| Starriver cloud text_render | M | `translator/ocr/dango.py` → `mangaRDR` | Optional typesetting backend. |
| Starriver translator (私人团子) | S | `translator/ocr/dango.py` → `dangoTrans` | New translator module. |
| Starriver font list | S | `mangaFontList` | For cloud render or font dropdown. |
| Dango v6 local manga API client | M | README v6.1.5; docs | Only if API is documented. |
| Translation history (DB + UI) | M–L | `utils/sqlite.py`, `ui/trans_history.py` | Extend context/glossary. |
| Filter/censor words | S | `ui/filter.py` | Post-process translator output. |
| Starriver quota display | S | `utils/http.py` → `onlineOCRQueryQuota` / `mangaOCRQueryQuota` | Need quota API URL. |

---

## 7. Implementable techniques from Dango code

Concrete algorithms and patterns from the [Dango-Translator Python repo](https://github.com/PantsuDango/Dango-Translator) that could be ported or adapted.

### 7.1 Text box merging (collision-based clustering)

**Where:** `utils/range.py`, `translator/ocr/dango.py` → `resultSortTD`, `resultSortMD`.

**Idea:** Merge OCR word/line boxes into text blocks by **axis-aligned rectangle collision** with a **language-dependent gap**:

- **Horizontal (横排) `resultSortTD`:**
  - Collision box = word box **expanded by 1.5× word height** vertically (so nearby lines merge).
  - `createRectangularTD`: box from word coordinates, height = word height + expansion.
  - `findRectangularTD`: recursive; find all words that collide with current box, merge into one block; compute union bbox and concatenate text (space between words for ENG).
- **Vertical (竖排) `resultSortMD`:**
  - Sort by x (right-to-left): `sort(key=lambda x: x["Coordinate"]["UpperRight"][0], reverse=True)`.
  - Collision box = word box **expanded by half word width** horizontally.
  - First pass: merge into column-like blocks. **Second pass** `findRectangular2MD`: merge those blocks horizontally (same `word_width` for column width) so multi-column text becomes one block per column; then sort by y (top-to-bottom) and output.

**BallonsTranslator today:** We have IoU-based merge (hf_object_det), `mit_merge_textlines` (easyocr_det), and font-size–based merge (ctd). Dango’s approach is **coordinate-only, no model** — good for **post-OCR** grouping of raw word boxes (e.g. from an API that returns words, not blocks). Useful if we add a “word-level OCR → merge into blocks” step or an optional merge mode for detector output.

**Implementation sketch:** Add a util (e.g. `utils/ocr_result_merge.py`) with:
- `merge_boxes_horizontal(boxes, text_list, gap_ratio=1.5)` and `merge_boxes_vertical(boxes, text_list, gap_ratio=0.5)` using AABB collision.
- Call after OCR when the engine returns word-level results; output `TextBlock` list.

---

### 7.2 OCR result sorting (horizontal vs vertical)

**Where:** `translator/ocr/dango.py` → `resultSortTD`, `resultSortMD`.

**Idea:**  
- **Horizontal:** Order blocks by reading order (e.g. top-to-bottom, then left-to-right); concatenate with `\n` between blocks; for ENG add space between words.  
- **Vertical:** Right-to-left columns, then top-to-bottom within column; optional second pass to group columns; output `\n` between blocks.

**BallonsTranslator today:** We have `sort_regions` and layout/reading-order logic in `textblock` / `text_layout`. Dango’s logic is tuned for their OCR format (Coordinate with UpperLeft/LowerRight etc.). We could adopt the **same ordering rules** (RTL sort, then TTB; collision-based grouping) for any OCR that returns word-level coordinates.

---

### 7.3 Image scaling for detection/API

**Where:** `translator/ocr/dango.py` → `imageDetect(image_base64, detect_scale)`.

**Formula:**
- Base: `longest_side = 1536` (same as our Starriver detector’s `short_side` for non–low-accuracy).
- If `detect_scale > 1`: `detect_scale = min(detect_scale * 0.6, 4)` then `longest_side = 1536 * detect_scale`.
- If `max(width, height) > longest_side`: scale so longest side = `longest_side`, keep aspect ratio; resize with `Image.ANTIALIAS`; re-encode to base64.

So they allow **1536–6144** (1536×4) effective longest side. Our Starriver detector uses a fixed short side (768 or 1536). We could add a **configurable “detection scale”** (e.g. 1–4) that sets `short_side = 1536 * scale` (capped) for API-based detectors to match Dango behavior and reduce failures on large pages.

---

### 7.4 Optional image border (padding)

**Where:** `translator/ocr/dango.py` → `imageBorder(src, dst, loc="a", width=3, color=(0,0,0))`.

**Idea:** Add a few pixels of padding (top/bottom/left/right or all) around the image before sending to OCR. Can help with edge-cut text.

**Implementation:** Optional “OCR crop padding” or “pre-OCR border” in our pipeline (e.g. in detector or OCR preprocess). Low priority; we already have crop padding per box.

---

### 7.5 UI / workflow ideas from `ui/manga.py`

| Feature | Dango | BallonsTranslator |
|--------|--------|--------------------|
| **One-click translate modes** | “跳过已翻译的” / “全部重新翻译” / “只重新翻译并渲染” / “只重新渲染” | We have pipeline toggles (detect/OCR/translate/inpaint); could add “Skip already translated” and “Re-translate only” / “Re-render only” as presets. |
| **Import** | From file, from folder, from multiple folders, from history path | We have folder/image input; “recent folders” and “multi-folder” could improve batch UX. |
| **Export** | Export to folder, export as ZIP, export project ZIP; “delete cache after export” | We have save project; ZIP export and “export then clean” would align with Dango. |
| **Tabs** | 原图 / 编辑 / 译图 (Original / Edit / Translated) with list + big preview | We have canvas and panel; “original vs edited vs translated” view toggles could help. |
| **Progress** | Dedicated progress bar for “import images” and “manga translate” | We have pipeline progress; optional per-phase (detect/OCR/translate/inpaint) progress is similar. |
| **File list sort** | `natsort.natsorted(files)` for natural order (e.g. 1, 2, 10 not 1, 10, 2) | We likely already sort; ensure natural sort for page order. |
| **Path length check** | Reject paths ≥250 chars to avoid Windows/temp issues | We could add a similar check and show a clear error. |

---

### 7.6 WebP → PNG before API

**Where:** `translator/ocr/dango.py` → `imageWebpToPng(filepath)`.

**Idea:** If the image is WebP, decode with PIL and re-encode as PNG before sending to the API (some backends expect JPEG/PNG).

**BallonsTranslator:** We could add the same in Starriver (and other API) detector/OCR when we have a file path or bytes; avoid “unsupported format” from the server.

---

### 7.7 Summary: what to implement first

| Priority | Item | Reason |
|----------|------|--------|
| **High** | Collision-based OCR box merging (7.1) | Reusable for any word-level OCR; improves block quality without changing models. |
| **High** | Detection scale param for API detectors (7.3) | Matches Dango; fewer 403/failures on large pages; single config knob. |
| **Medium** | “Skip already translated” / “Re-translate only” (7.5) | Speeds up re-runs and re-renders. |
| **Medium** | Export as ZIP / export then clean (7.5) | Common batch workflow. |
| **Low** | WebP→PNG before API (7.6) | Avoids server rejections. |
| **Low** | Pre-OCR border (7.4) | Minor; we have crop padding. |

---

## 8. References

- **Dango-Translator (GitHub):** [PantsuDango/Dango-Translator](https://github.com/PantsuDango/Dango-Translator) — legacy Python (v4.5.8); current app is Golang v6.x.
- **Dango docs:** [dango-docs-v6.ap-sh.starivercs.cn](https://dango-docs-v6.ap-sh.starivercs.cn/) — use guide; may document local API.
- **Download (register in app):** [translator.dango.cloud](https://translator.dango.cloud/).
- **Starriver dashboard / login:** [dashboard.stariver.org.cn](https://dashboard.stariver.org.cn/).
- **BallonsTranslator Starriver usage:** [STARRIVER.md](STARRIVER.md).
