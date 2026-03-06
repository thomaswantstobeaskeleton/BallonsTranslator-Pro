# Related manga/translation projects

Short reference for how BallonsTranslator relates to other open-source manga/comic translation tools and what was adopted from them.

## manga-image-translator (zyddnys)

**Repo:** [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator)

One-click translation of text in manga/images: detection → OCR → translation → inpainting → rendering. Supports Japanese, Chinese, English, and many others. Detectors: default, dbconvnext, **CTD**, CRAFT, Paddle. Config includes `detect_size`, `text_threshold`, `box_threshold`, `unclip_ratio`, `mask_dilation_offset`, and pre-processing: **det_rotate**, **det_invert**, **det_gamma_correct**, **det_auto_rotate**.

### What BallonsTranslator took from it

- **CTD (ComicTextDetector)** is the same family: we use the same model files (`comictextdetector.pt` / `.onnx`) from their releases and the same CTD pipeline (detection size, box threshold, merge, mask dilation).
- **CTD options aligned with manga-image-translator:** In the **ctd** detector we added manga-image-translator-style pre-processing so you can match their behavior:
  - **det_invert** – invert image before detection (helps light text on dark).
  - **det_gamma_correct** – gamma correction before detection.
  - **det_rotate** – rotate image 90° before detection, then rotate results back (for mainly vertical text).
  - **det_auto_rotate** – if majority of detected text is horizontal, re-run detection with 90° rotation.
  - **det_min_image_side** – add border so smallest side is at least N px (e.g. 400), then crop back (for small images).

We did not port **panel_finder**; you can combine **ctd** with **det_rotate** or **det_auto_rotate** when needed.

---

## manga-translator-ui (hgmzhn)

**Repo:** [hgmzhn/manga-translator-ui](https://github.com/hgmzhn/manga-translator-ui)

A **Qt desktop + Web + CLI** wrapper around the **manga-image-translator** core. Same pipeline (detection → OCR → translation → inpainting → rendering). Features: 5 translation engines (OpenAI, Gemini normal/HQ, Sakura), **AI line-break** (hint number of regions to the model), **automatic glossary extraction**, PSD export, **replace translation** mode (extract from translated image, apply to raw), and workflows: export original, import translation and render, colorize only, upscale only, inpaint only.

### What BallonsTranslator took from it

- **AI line-break hint:** In **LLM_API_Translator**, an optional parameter **Hint original regions** (manga-translator-ui style). When enabled, the user prompt is prefixed with `[Original regions: N]` so the model knows to output exactly N translations and can break lines per region. Improves 1:1 region-to-translation behavior without extra API calls.
- **check_br_and_retry:** When translation count mismatches source count, we retry the request up to **invalid repeat count** (same idea as their [BR] retry).
- **Post-translation check:** Optional validation (repetition threshold, target-language ratio) with **post_translation_check**, **post_check_repetition_chars**, **post_check_target_ratio**, **post_check_max_retries**; retries the API call on failure.
- **Glossary auto-extract:** **extract_glossary** runs one extra LLM call after each batch to extract terms and append to the series glossary (series_context_path).
- **Replace translation mode:** **replace_translation_mode** + **replace_translation_translated_dir**: load translated image (same filename in a folder), detect+OCR on it, match blocks by IoU, set blk.translation, OCR raw for source, then inpaint and render.
- **center_text_in_bubble** and **optimize_line_breaks:** Module options for layout (center text in bubble; allow slightly more aggressive scale-up for fewer lines).
- **Welcome / first-start screen:** When no project is open at startup, a welcome screen is shown (config: **Show welcome screen when no project is open** in General). Inspired by [manhua-translator](https://github.com/aakaka525-design/manhua-translator) and [Komakun](https://github.com/drawhisper-org/komakun): open folder, open project, open images, open ACBF/CBZ, and a list of recent projects in a single bubbly view.
- **Research doc:** Deep feature comparison and implementation status are in **docs/RELATED_PROJECTS_RESEARCH.md** (workflows, config, etc.).

### What we did not implement (reference only)

- **PSD export** – We have other export options; PSD not added.

---

## yahao333/manga_translator

**Repo:** [yahao333/manga_translator](https://github.com/yahao333/manga_translator)

A **Next.js/TypeScript** frontend (Vercel): landing page, multi-language (ZH/EN/JA), responsive. No Python detection/OCR/translation pipeline in the repo—it’s a UI/product shell. There is no detector or OCR code to port into BallonsTranslator; only the product idea (multi-language manga translation) is comparable.

---

## Komakun (drawhisper-org)

**Repo:** [drawhisper-org/komakun](https://github.com/drawhisper-org/komakun)

**KomaKun!** is a **browser-based manga translation IDE**: import pages → AI OCR (Google Cloud Vision) → neural inpainting (Replicate) → LLM translation (Replicate, OpenAI, OpenRouter, local) → typesetting (react-konva). No local detector/OCR models; everything goes through APIs (Vision for OCR, Replicate for inpainting + LLM).

### Comparison with BallonsTranslator

| Aspect        | Komakun                         | BallonsTranslator                          |
|---------------|----------------------------------|--------------------------------------------|
| **Detection** | Google Cloud Vision (API)        | Local detectors (CTD, Paddle, Surya, …)    |
| **OCR**       | Vision API                       | Local OCRs (Paddle, Surya, Manga OCR, …)   |
| **Inpainting**| Replicate (cloud)               | Local (LaMa, AOT, FLUX, …)                  |
| **Translation** | Replicate / OpenAI / OpenRouter / local | Many (Google, DeepL, Sakura, LLM API, …) |
| **Stack**     | Next.js, browser-only           | PySide, desktop                             |

We did not port any Komakun code; the architecture is different (API-first vs local-first). Use Komakun when you want a zero-install browser workflow and cloud APIs; use BallonsTranslator when you want local models and a desktop pipeline.

---

## Summary

- **manga-image-translator:** Same CTD model and pipeline; we added **det_invert**, **det_gamma_correct**, **det_rotate** to **ctd** to align with their detection options.
- **manga-translator-ui (hgmzhn):** We added **Hint original regions**, **check_br_and_retry**, **post-translation check**, **extract_glossary**, **replace_translation_mode**, **center_text_in_bubble**, **optimize_line_breaks**, and documented workflows/config in **RELATED_PROJECTS_RESEARCH.md**.
- **yahao333/manga_translator:** Frontend-only; no detector/OCR to integrate.
- **Komakun:** Browser IDE with cloud APIs; different stack; no code ported.
