# Possible Future Additions for BallonsTranslator Pro

Research-backed list of features that could be implemented next. Sources: upstream GitHub issues, in-repo TODOs, `docs/SETTINGS_UI_RECOMMENDATIONS.md`, `docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md`, and comparable manga translation tools.

---

## Implemented (this codebase)

| Feature | Where |
|---------|--------|
| **Batch retry/skip/terminate** on translation failure | When a page fails during the translate pipeline (single-page or batch queue), a dialog offers **Retry**, **Skip**, or **Terminate**. Retry re-queues the page; Skip continues without it; Terminate clears the queue and stops. `ui/module_manager.py`: `TranslateThread._run_translate_pipeline`, `ModuleManager.on_translation_failure_request`, `translation_failure_request` signal, `_trans_failure_mutex` / `_trans_failure_condition`. |
| **Apply to all blocks** when no block selected | Already works: the "Apply to all blocks" button applies the current global format to every text block on the page; selection is not required. |
| **Export current page as** (JPG/PNG/WebP/JXL) | File / Export current page as... saves the current page result image. Use after running the pipeline (or when a result file exists). For saving after manual inpainting without running translation, use Ctrl+S; result is also auto-saved on page switch/close. |
| **.gitignore** | Added `*.pyc`. Added `docs/PROMPT_FIND_MANGA_DOWNLOAD_SOURCES.md` (maintainer-only prompt for finding manga download sources). |
| **Global stroke size/color in Config** | Config → General → Typesetting: **Default stroke width** (0–5, step 0.1) and **Default stroke color** (color button). These set `pcfg.global_fontformat.stroke_width` and `.srgb`; used when "use global setting" is selected for Stroke Size / Stroke Color. Upstream #1143. |
| **Project glossary UI** | Already present: **Edit → Translation context (project)...** (or Config → Translator → **Translation context (project)...**) opens a dialog with Series context path and **Project glossary** (one line per entry: `source -> target`). Saved to project JSON. |
| **Export current page when no result file (#1134)** | **Export current page as** now falls back to inpainted image (file or in-memory) and then to original image, so users can save after manual inpainting without running the full pipeline. The success message indicates what was saved (result / inpainted / original). |
| **Next-page context for translation (#1142)** | LLM_API_Translator has a **context_next_page** checkbox (Config → Translator). When enabled, the next page's source text (up to 5 lines, 500 chars) is included in the prompt as "Next page (for context): …" for continuity. Previous-page context unchanged. |
| **Check and download module files** | **BaseModule.load_model()** runs check/download for the module’s `download_file_list` (via `download_and_check_module_files([cls])`) before loading. Config → DL Module has **Check / Download module files**; it runs check/download for all detectors, OCR, inpainters, and translators and shows a message. `modules/base.py`, `ui/configpanel.py`. |
| **ChatGPT translation context (previous/next page)** | ChatGPT translator supports **context previous pages count** (0–5), **context next page** (checkbox), and **context max chars** (Config → Translator). When set, previous-page translations and next-page source lines are prepended to the prompt for terminology/style continuity. Uses `set_translation_context`; same pipeline as LLM_API. `modules/translators/trans_chatgpt.py`. |

---

## 1. From upstream (dmMaze/BallonsTranslator) open issues

| Issue | Request | Pro status / note |
|-------|---------|-------------------|
| **#1143** | Stroke size and color in **global** settings | Stroke exists per-block; global default would match "save as default" pattern. |
| **#1142** | **Local VLM + more context** for translation | **Partial:** Next-page context added (Config → Translator: **context_next_page**). Previous-page context and glossary already present. Full local VLM or chapter summary would be a separate addition. |
| **#1141** | **ACBF format** and **CBZ multilingual** support | Comic book formats; would allow export/import of multi-language comics in standard formats. |
| **#1134** | **Export/save as JPG or PNG** after manual inpaint+edit **without** running translation | **Implemented.** Export current page as uses result → inpainted → original fallback; message shows what was saved. |
| **#1131** | Use **VL (vision-language) model** instead of separate detection + OCR | Single model for detection+recognition; could simplify pipeline and improve quality on some content. |
| **#891** | Configurable **padding after CTD** | Pro already has configurable box padding for CTD and others; could be referenced in upstream. |

---

## 2. From this repo (TODOs and docs)

| Location | Idea | Notes |
|----------|------|--------|
| **TRANSLATION_CONTEXT_AND_GLOSSARY.md** | **Project glossary UI** | **Already implemented:** Edit → Translation context (project)... dialog has Project glossary (plain text, `source -> target` per line); saved to project JSON. |
| **TRANSLATION_CONTEXT_AND_GLOSSARY.md** | **Document-level translation** | Translate full work as one document then map back to pages for max consistency. Doc: "Token limits, memory, and the current app design (page-by-page pipeline) make this a larger change. Not implemented here; left as a future option." |
| **ui/module_manager.py** | **Retry / skip / terminate** on translation failure in batch | **Implemented.** When translation fails during pipeline, user gets Retry / Skip / Terminate dialog. |
| **modules/base.py** | **Check and download files** & inform UIs | **Implemented.** `load_model()` runs `download_and_check_module_files([cls])` before `_load_model()`. Config has "Check / Download module files" button; failures are logged and the dialog points users to the console. |
| **modules/translators/trans_chatgpt.py** | **Summarizations as context** | **Implemented (context):** Previous/next page context added (context previous pages count, context next page, context max chars). Optional future: per-chunk summarization when prompt is split (extra API call). |
| **modules/textdetector/base.py** | **Full-project detection with progress** | TODO: "allow processing proj entirely in _detect and yield progress" for better progress reporting on large projects. |
| **modules/__init__.py** | **manga-image-translator as backend** | TODO: "use manga-image-translator as backend" — potential alternative pipeline or modules. |

---

## 3. From SETTINGS_UI_RECOMMENDATIONS.md

The doc marks several items as "Possible additions (not yet implemented)"; many are **already implemented** in Pro (e.g. font weight, quick opacity, "Apply to all blocks", "Save as default", strikethrough, gradient type, tool hints, Rect labels, Erode, Fill without model, inpaint brush hardness, pen alpha). Remaining or reinforcing ideas:

- **Apply global format when no block selected** — **Already supported:** "Apply to all blocks" applies to every block on the page; no selection needed.
- **Clearer "manual save" behavior** — Ctrl+S and "Export current page as..." exist. Optional future: save current canvas state when no result file exists (e.g. after manual inpainting only).

---

## 4. From other manga translation tools (2024–2025)

Common features that could inspire Pro:

- **Richer inpainting targets** — e.g. mosaic/overlay removal, black bars (KitsuTL-style "unmask").
- **Very large language coverage** — e.g. 120+ languages (Pro already supports many via existing translators).
- **Batch + robustness** — Set-and-forget batch with clear pause/resume/cancel and **per-page retry/skip** (aligns with §2).
- **Context and quality** — Chapter/volume summaries and previous+next page context (aligns with #1142 and trans_chatgpt TODO).
- **Formats** — ACBF/CBZ and other comic standards (aligns with #1141).
- **VL-based pipeline** — Single model for detection+OCR (aligns with #1131).

---

## 5. Prioritized shortlist (suggested)

1. ~~**Batch retry/skip/terminate**~~ — **Done.** See "Implemented" above.
2. ~~**Project glossary tab / UI**~~ — **Done.** Translation context (project) dialog has project glossary; see "Implemented" above.
3. ~~**Global stroke size/color in Config**~~ — **Done.** Config → General → Typesetting: Default stroke width + Default stroke color. See "Implemented" above.
4. ~~**Local VLM + more context**~~ — **Partial.** Next-page context added (context_next_page in LLM translator). See "Implemented" above.
5. ~~**Export/save without translation**~~ — **Done.** Export current page as uses result → inpainted → original fallback. See "Implemented" above.
6. **ACBF / CBZ multilingual** — Upstream #1141; useful for interoperable comic workflows.
7. **VL model as OCR/detection** — Upstream #1131; larger change, potential quality gain.
8. **Document-level translation** — Large design change; good long-term target.
9. **manga-image-translator backend** — Optional alternative pipeline; depends on maintainability and fit.

---

## 6. References

- Upstream issues: [dmMaze/BallonsTranslator issues](https://github.com/dmMaze/BallonsTranslator/issues)
- Context/glossary design: `docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md`
- UI/settings roadmap: `docs/SETTINGS_UI_RECOMMENDATIONS.md`
- Multimodal manga translation: [plippmann/multimodal-manga-translation](https://github.com/plippmann/multimodal-manga-translation)
