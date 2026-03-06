# Related projects – deep research

Consolidated research from [manga-image-translator](https://github.com/zyddnys/manga-image-translator), [yahao333/manga_translator](https://github.com/yahao333/manga_translator), [drawhisper-org/komakun](https://github.com/drawhisper-org/komakun), and [hgmzhn/manga-translator-ui](https://github.com/hgmzhn/manga-translator-ui). Use this doc to keep feature parity and implementation status consistent.

---

## 1. manga-image-translator (zyddnys)

**Repo:** https://github.com/zyddnys/manga-image-translator  
**Stack:** Python, detection → OCR → translation → inpainting → rendering. CLI + Web + API.

### Detection

| Feature | Config / param | BallonsTranslator status |
|--------|----------------|---------------------------|
| Detectors | default, dbconvnext, **ctd**, craft, paddle, none | **ctd** same model; we added det_invert, det_gamma_correct, det_rotate |
| detect_size | int (e.g. 2048) | ✅ ctd: detect_size selector |
| text_threshold | float | ✅ (craft/easyocr style); ctd uses box score threshold |
| box_threshold | float | ✅ ctd: box score threshold |
| unclip_ratio | float | In db_utils (SegDetectorRepresenter); not exposed at ctd top level |
| mask_dilation_offset | int | ✅ ctd: mask dilate size |
| det_rotate, det_invert, det_gamma_correct, det_auto_rotate | bool | ✅ ctd: all four (det_auto_rotate re-runs with 90° when majority horizontal) |
| add_border (small images) | minimum_image_size 400 | ✅ ctd: det_min_image_side (0=off, 400=add border then crop back) |

### Translation / render

- Translator chain: `trans1:lang1;trans2:lang2`. We have translator_chain in config (manga-translator-ui).
- Render: default, manga2eng, none; alignment; direction; font_size_offset; no_hyphenation; font_color; line_spacing; etc. We have similar in fontformat/layout.

### Implemented in BallonsTranslator

- CTD from same releases; det_invert, det_gamma_correct, det_rotate on **ctd** detector.

---

## 2. yahao333/manga_translator

**Repo:** https://github.com/yahao333/manga_translator  
**Stack:** Next.js, TypeScript, Vercel. Landing page + multi-language (ZH/EN/JA). No backend detection/OCR/translation in repo.

**Conclusion:** No detector/OCR/pipeline code to port. Product idea only.

---

## 3. Komakun (drawhisper-org)

**Repo:** https://github.com/drawhisper-org/komakun  
**Stack:** Next.js, react-konva, Google Cloud Vision (OCR), Replicate (inpainting + LLM). Browser-only, API-first.

| Aspect | Komakun | BallonsTranslator |
|--------|---------|-------------------|
| Detection | Google Cloud Vision API | Local (CTD, Paddle, Surya, …) |
| OCR | Vision API | Local OCRs |
| Inpainting | Replicate | Local (LaMa, AOT, …) |
| Translation | Replicate / OpenAI / OpenRouter / local | Many (Google, DeepL, Sakura, LLM API, …) |

**Conclusion:** Different architecture; no code ported. Documented in RELATED_PROJECTS.md.

---

## 4. manga-translator-ui (hgmzhn)

**Repo:** https://github.com/hgmzhn/manga-translator-ui  
**Stack:** Based on **manga-image-translator** core; Qt desktop UI + Web + CLI. Same pipeline: detection → OCR → translation → inpainting → rendering.

### Workflows (from doc/WORKFLOWS.md)

| Workflow | Description | BallonsTranslator equivalent |
|----------|-------------|------------------------------|
| 正常翻译 | Full translate | ✅ Run pipeline |
| 导出翻译 | Translate + export TXT/JSON | ✅ We have project JSON; export script possible |
| 导出原文 | Detect + OCR only, export original TXT/JSON | ✅ Detect+OCR then export/save text |
| 导入翻译并渲染 | Load TXT/JSON translation, render only | ✅ Load project / paste translation |
| 仅上色 | Colorize only | ❌ We don’t have colorizer module |
| 仅超分 | Upscale only | ❌ Optional upscale step in pipeline possible |
| 仅修复 | Inpaint only (erase text) | ✅ Can run detect → inpaint without translate |
| 替换翻译 | Extract text from translated image, apply to raw (template alignment) | ✅ replace_translation_mode + replace_translation_translated_dir; detect+OCR on translated, match by IoU, set blk.translation, OCR raw for source, then inpaint |

### Config highlights (from config.py / SETTINGS.md)

**Render**

- layout_mode: smart_scaling, strict, balloon_fill, disable_all. We have layout/font scaling.
- center_text_in_bubble, optimize_line_breaks, check_br_and_retry, strict_smart_scaling.
- stroke_width (ratio), enable_template_alignment, paste_mask_dilation_pixels (replace mode).
- force_strict_layout, disable_auto_wrap.

**Translator**

- openai, openai_hq, gemini, gemini_hq, sakura, none, original.
- extract_glossary: auto-extract terms to prompt glossary (Person, Location, Org, Item, Skill, Creature). We have translation_glossary and series glossary; no auto-extract.
- high_quality_prompt_path, translator_chain, selective_translation.
- enable_post_translation_check, post_check_max_retry_attempts, post_check_repetition_threshold, post_check_target_lang_threshold. We have invalid repeat count; no full “post-check” with repetition/lang ratio.

**Detection**

- use_yolo_obb, yolo_obb_conf, yolo_obb_overlap_threshold, min_box_area_ratio. We have hf_object_det / ysgyolo; different param names.

**OCR**

- use_hybrid_ocr, secondary_ocr; ignore_bubble; use_model_bubble_filter, model_bubble_overlap_threshold; merge_gamma, merge_sigma, merge_edge_ratio_threshold, merge_special_require_full_wrap; ocr_vl_language_hint, ocr_vl_custom_prompt. We have various OCRs; some merge/ignore options exist elsewhere.

**CLI / export**

- export_editable_psd, psd_font, psd_script_only. We have GIMP/PSD export in some code paths.
- replace_translation mode.

### AI 断句 (AI line break) – manga-translator-ui

- In translation request, prefix with **`[Original regions: N]`** so the model knows there are N text regions and can break lines accordingly.
- Supported for OpenAI, Gemini (including HQ). No extra API call.
- **Implemented in BallonsTranslator:** Optional “hint original regions” in LLM_API_Translator: when enabled, prepend `[Original regions: N]` to the user prompt (see trans_llm_api.py).

### 自动提取术语 (extract_glossary)

- After translation, call model to extract terms (Person, Location, Org, etc.) and append to prompt file’s glossary. Requires high_quality_prompt_path.
- **Status:** Not implemented in BallonsTranslator. Could be a post-step calling same LLM with a glossary-extraction prompt.

### Implemented / to implement

| Feature | Status |
|---------|--------|
| AI line-break hint (original regions) in LLM prompt | ✅ Added to LLM_API_Translator |
| Glossary auto-extract | ❌ Doc only; optional future |
| Replace translation mode | ❌ Doc only |
| center_text_in_bubble, optimize_line_breaks | Partial (we have layout/scale); explicit center_text_in_bubble option possible |
| Post-translation check (repetition, target lang ratio) | Partial (retry on count mismatch); full check optional |
| PSD export (editable layers) | We have GIMP/PSD in some flows; parity doc |

---

## 5. Feature matrix (all projects)

| Feature | zyddnys | yahao333 | komakun | hgmzhn | BallonsTranslator |
|---------|---------|----------|---------|--------|-------------------|
| Local text detection | ✅ | ❌ | ❌ | ✅ | ✅ |
| CTD + det preprocess | ✅ | ❌ | ❌ | ✅ | ✅ |
| Multiple detectors | ✅ | ❌ | ❌ | ✅ | ✅ (many) |
| Local OCR | ✅ | ❌ | ❌ | ✅ | ✅ |
| API OCR (e.g. Vision) | ❌ | ❌ | ✅ | ❌ | Optional (e.g. Stariver) |
| Local inpainting | ✅ | ❌ | ❌ | ✅ | ✅ |
| Cloud inpainting | ❌ | ❌ | ✅ | ❌ | ❌ |
| Translator chain | ✅ | ❌ | ❌ | ✅ | Via config/ensemble |
| Glossary (manual) | ✅ | ❌ | ❌ | ✅ | ✅ (glossary + series) |
| Glossary auto-extract | ❌ | ❌ | ❌ | ✅ | ❌ |
| AI line-break hint (regions) | ❌ | ❌ | ❌ | ✅ | ✅ (LLM_API_Translator) |
| Export original / import translate | ✅ | ❌ | ❌ | ✅ | ✅ (project load/save) |
| PSD export | ✅ (GIMP) | ❌ | ❌ | ✅ (PSD) | GIMP/PSD in places |
| Replace translation mode | ❌ | ❌ | ❌ | ✅ | ❌ |
| Colorize only | ✅ | ❌ | ❌ | ✅ | ❌ |
| Upscale only | ✅ | ❌ | ❌ | ✅ | Optional in pipeline |
| Inpaint only | ✅ | ❌ | ❌ | ✅ | ✅ (skip translate) |

---

## 6. References

- [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator) – detection/, config, common.py
- [yahao333/manga_translator](https://github.com/yahao333/manga_translator) – Next.js front
- [drawhisper-org/komakun](https://github.com/drawhisper-org/komakun) – README, stack
- [hgmzhn/manga-translator-ui](https://github.com/hgmzhn/manga-translator-ui) – config.py, save.py, doc/WORKFLOWS.md, doc/FEATURES.md, doc/SETTINGS.md

See also **RELATED_PROJECTS.md** for user-facing summary and **EASYSCANLATE_INTEGRATION.md** for EasyScanlate parity.
