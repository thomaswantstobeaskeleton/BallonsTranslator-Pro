# Settings UI: recommendations and roadmap

Suggestions for **General** and **DL Module** settings (and related UI) that are **not** model-specific. Some ideas come from the original BallonsTranslator GitHub issues. **BallonsTranslatorPro** has implemented the items below.

---

## Implemented in BallonsTranslatorPro

### General → Startup
- **Logical DPI** — Spinbox (0 = system, 96/72 typical). Persisted in config; applied on next launch via `launch.py`.
- **Recent projects limit** — Spinbox 5–30 (default 14). Recent list is trimmed when loading and when updating.
- **Confirm before Run** — Checkbox. When unchecked, Run no longer shows the Run/Continue/Cancel dialog.
- **Dark mode** — Checkbox. Synced with View → Dark Mode; changing either updates the other and reapplies theme.
- **Display language** — Dropdown (English / 简体中文 / etc.). Same as View → Display Language; toggles UI language without restart.
- **Config panel font scale** — Spinbox 0.8–1.5. Scales font size in the Config panel for accessibility.
- **Spell check OCR result** — Checkbox under **General → OCR result**: "Spell check / Auto-correct OCR result". When enabled, after OCR runs, each word is checked with a spell checker (pyenchant); if a word is misspelled and there is exactly one suggestion, it is replaced (e.g. "teh" → "the"). Requires `pyenchant` and a system dictionary (e.g. en_US). This is the **auto-correct** for OCR text.

### General → Typesetting
- **Text in box** — Dropdown ([#1077](https://github.com/dmMaze/BallonsTranslator/issues/1077)): **Auto fit to box** (program scales font size so text fits the balloon, line structure unchanged) or **Fixed size (use font size list)**. Syncs with Font Size ("decide by program") and **Auto layout** so one control sets both. More font sizes in the list (30, 32, 34, 40, 44).

### DL Module
- **Default device** — Dropdown: "Default (use module default)" or a specific device (CPU, CUDA, etc.). Used when a module’s device param is set to "Default". Empty device params are filled from this when loading config.
- **Run pipeline presets** — Run menu → Pipeline presets: "Full", "Detect + OCR only", "Translate only", "Inpaint only". Apply and sync the four stage toggles.
- **OCR font detection** — OCR section: "Font Detection" checkbox with tooltip (detect font properties from image after OCR).
- **Unload after idle** — Spinbox (0–120 minutes, 0 = off). Unloads DL models after N minutes with no Run or pipeline activity; timer resets when pipeline finishes. *Note: With this or "Load model on demand" enabled, **Test translator** may fail sometimes ("No translator loaded") because the translator is only loaded when you run a pipeline — run a page first or temporarily disable those options when testing.*

### Tools menu
- **Batch queue (#1020)** — **Tools → Batch queue...** opens a dialog to add multiple folders to a queue and run the pipeline on each in sequence (same behavior as headless `--exec_dirs`). **Add folder(s)...** adds one folder; **Add folder (include subfolders)** adds the selected folder and each of its immediate subfolders as separate items. **Pause** / **Resume** temporarily halt the pipeline; **Cancel queue** stops the current job and clears the remaining queue. The list shrinks as each folder is processed.
- **Re-run detection only** — Runs pipeline with only detection enabled; restores previous stage flags when done.
- **Re-run OCR only** — Runs pipeline with only OCR enabled (re-recognize text, keep boxes); restores previous stage flags when done.
- **Export all pages** — Exports all result images to a chosen folder (pages that have no result yet are reported as missing).
- **Check project** — Validates project: missing image files, invalid project JSON; shows a report dialog.
- **Manga / Comic source...** — Opens a dialog to search for manga/manhua/manhwa titles (via MangaDex API), list chapters by language, select chapters to download, and optionally open the downloaded chapter folder in BallonsTranslator to translate. Supports “data-saver” (smaller) or original quality. **Sources:** MangaDex (search by title) and MangaDex by chapter URL (paste a single chapter link). **Downloaded pages are saved as 001.ext, 002.ext, …** so BallonsTranslator loads them in the correct order. **Config:** Language, data-saver, download folder, and request delay (rate limiting) are persisted and restored. Optional “Request delay” (0–2 s) throttles API requests.

### Other
- **Config safety:** API keys removed from source; config.example.json and GITHUB_UPLOAD.md added.
- **Lossless WebP (#1055)** — In Config → General → Save: when result format is WebP, a "WebP lossless" checkbox enables lossless encoding (quality setting ignored). Export all pages and normal Save both respect this option.

### Canvas right-click menu (text edit mode)
- **Detect text in region** — Right-drag to draw a rectangle, then right-click; choose "Detect text in region" to run the text detector on that area only and add new text blocks (addresses upstream #1137 manual text detection).
- **Detect text on page** — When no region is selected, runs text detection on the full page (convenience alternative to Run pipeline with only detection).
- **Merge selected blocks** — Merges source and translation of all selected text blocks into the first block (by index), then removes the others.
- **Move block(s) up** / **Move block(s) down** — Move the selected block one position up or down in the block list (and in the text panel). Single-block selection only.
- **Copy / Paste translation** — Copy or paste translation text for selected blocks (in addition to existing Copy/Paste source text).
- **Clear source text** / **Clear translation** — Clear source or translation for selected blocks.
- **Select all** — Select all text blocks on the page.
- **Spell check source text** / **Spell check translation** — Run spell check / auto-correct on source or translation of selected blocks (uses pyenchant; requires pyenchant and a system dictionary; shows a warning if unavailable).
- **Trim whitespace** — Remove leading and trailing whitespace from each line (source and translation) in selected blocks.
- **To uppercase** / **To lowercase** — Convert source and translation text in selected blocks to uppercase or lowercase.
- **Toggle strikethrough** — Toggle strikethrough on selected blocks (same as the format panel button).
- **Gradient type** — Submenu: **Linear** or **Radial** to set gradient type on selected blocks.
- **Text on path** ([#1138](https://github.com/dmMaze/BallonsTranslator/issues/1138)) — Submenu: **None**, **Circular**, or **Arc** to draw text along a circle or arc on selected blocks (for balloons and SFX).
- Existing: Copy, Paste, Delete, Copy/Paste source text, Delete and Recover, Apply font formatting, Auto layout, Reset Angle, Squeeze, Translate, OCR, OCR and translate, OCR+translate+inpaint, Inpaint.

---

## Font format & Inpaint / Drawing panel

### Font format (global + advanced)

Implemented in this codebase: **Quick opacity** in main row (Opacity label + combobox); **Apply to all blocks** button; **Save as default** button (writes global format to config); **Font weight** in Advanced panel (Light/Normal/Medium/SemiBold/Bold). **Strikethrough** in main format row (button + right-click "Toggle strikethrough"). **Gradient type** (Linear / Radial) in Advanced gradient group + right-click "Gradient type" submenu. **Text on path** ([#1138](https://github.com/dmMaze/BallonsTranslator/issues/1138)): format panel dropdown **Text on path** (None / Circular / Arc) and **Arc degrees** spinbox when Arc is selected; right-click canvas submenu **Text on path → None / Circular / Arc**. Circular draws text along a full circle; Arc uses the arc span (e.g. 180°). **Text warp** ([#1093](https://github.com/dmMaze/BallonsTranslator/issues/1093)): format panel **Warp** dropdown (None / Arc / Arch / Bulge / Flag) and **Warp strength** (0.1–1) for Photoshop-like distortion. **Text eraser** ([#1093](https://github.com/dmMaze/BallonsTranslator/issues/1093)): Drawing panel **Text eraser** tool — select text block(s), choose Text eraser, draw on canvas to erase parts of the text (creates depth effect so text appears behind objects). Rect panel: **Fill** button (fill with background, no model), **Erode** slider, descriptive method names (Canny + flood, etc.), tool shortcut hints in tooltips; **pen alpha** persisted (`pentool_color` [r,g,b,a]); **Inpaint brush hardness** slider (0–100, soft to hard edge).

**Current:** Global format (typesetting panel): font family, size, line/letter spacing, opacity (quick), color, alignment, bold/italic/underline/**strikethrough**, vertical, stroke width/color. Advanced: line spacing type, opacity (full), font weight, shadow, **gradient type (Linear/Radial)**, gradient start/end color, angle, size. Style presets for named formats.

**Possible additions (not yet implemented):**

- **Font weight (100–900)** — `FontFormat` and `ffmt_change_font_weight` exist; only Bold (Normal/Bold) is in the main panel. Add a **Font weight** combo in the **Advanced** panel (e.g. Light 300 / Normal 400 / Medium 500 / SemiBold 600 / Bold 700) for finer control.
- **Quick opacity in main row** — Opacity is only in Advanced; an optional compact opacity control in the main format row for quick access.
- **Apply global format to all blocks** — Button or action: “Apply to all blocks on page” to apply the current global font format to every text block on the current page (convenience when no block is selected).
- **Set global format as default** — “Save as default” to overwrite the saved default global format (e.g. in config) so new projects start with this format.
- **Strikethrough** — Add to `FontFormat` and to the format/advanced UI if desired.
- **Gradient type** — If the renderer supports it, add linear vs radial in the Advanced gradient group.

### Inpaint / Drawing panel (right-side, 4 tools)

Implemented in this codebase: Tool shortcut hints (Hand H, Inpaint J, Pen B, Rect R) in tooltips; Rect method labels (Canny + flood, Connected Canny, Use existing mask); Erode slider; **Fill** button (fill with detected background, no inpainter); pen alpha persisted in config; Dilate/Erode tooltips; **Inpaint brush hardness** slider (0–100, feathered to hard edge).

**Current:** **Hand** (pan), **Inpaint** (brush: thickness, shape circle/rect, inpainter selector), **Pen** (thickness, shape, color, alpha), **Rect** (dilate, method 1/2/Use existing mask, Auto, Inpaint/Delete). Shared: Mask opacity slider; inpainter chosen from config panel when the combobox is shown in the stack.

**Possible additions (not yet implemented):**

- **Tool shortcut hints** — Show keyboard shortcuts in tool tips or labels: Hand **H**, Inpaint **J**, Pen **B**, Rect **R**.
- **Rect method labels** — Replace “method 1” / “method 2” with descriptive names, e.g. “Canny + flood”, “Connected Canny”, “Use existing mask”.
- **Rect: Erode slider** — In addition to Dilate, an optional **Erode** slider to shrink the mask (useful when the segmentation is too large).
- **Rect: Fill without model** — Option “Fill with background” or “Fill with color” to fill the selection without running the inpainter (faster for solid areas).
- **Inpaint brush hardness** — Optional brush hardness/feathering for the inpaint brush (so mask edges are soft); would require pipeline support.
- **Persist pen alpha** — Ensure pen tool alpha is saved/loaded in `DrawPanelConfig` (e.g. 4-component `pentool_color` or separate `pentool_alpha`).

---

## Possible future additions

- **Batch export to PDF** — Option in Export all pages to also build a single PDF from the exported images.
- **Duplicate/overlapping block check** — In "Check project", optionally report duplicate or overlapping text blocks.
- **More manga sources** — Add more sources to the Manga / Comic source dialog (e.g. other APIs or scrapers) alongside MangaDex.

---

## From upstream issues

- **#591 Spell checking for OCR** — Implemented via optional pyenchant + Config checkbox (auto-correct after OCR). **Manual spell check** is also available: right-click on selected blocks → **Spell check source text** or **Spell check translation** (same engine; works on already-edited text).
- **#1055 Lossless WebP** — Implemented via Config "WebP lossless" checkbox when result format is WebP.
- **#1077 Text in box / font sizes** — **Text in box** dropdown in Config → General → Typesetting: "Auto fit to box" (program decides font size so text fits the balloon; keeps line structure) or "Fixed size (use font size list)". More font sizes in the list (30, 32, 34, 40, 44). Drag-and-drop of folder/images onto canvas to open project (if unavailable, use File → Open).
- **#867, #854** — Addressed by this fork’s extra detectors and Paddle v5.
- **#685 / #554 Google OCR** — Fork uses params for api_key (no hardcoded keys).
- **#733 Inpainting gray** — See MANHUA_BEST_SETTINGS.md (inpaint size, mask dilation).

---

## Where to edit

- **Config panel:** `ui/configpanel.py` — checkboxes, combos, spinboxes; connect to `pcfg` in `utils/config.py`.
- **Config schema:** `utils/config.py` — add fields to `ProgramConfig` or `ModuleConfig`.
- **Tools / Run menu:** `ui/mainwindowbars.py` — add `QAction`s; handlers in `ui/mainwindow.py`.

This doc can be updated as more items are implemented or as community feedback comes in.
