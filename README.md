# BallonsTranslator – Extended Modifications (README)

This document describes **everything modified or added** in this fork compared to the original [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator): **all new models** (detection, OCR, inpainting, translation), **how to run each**, **settings and tuning** (inpaint sizing, mask dilation/kernel, text/box formatting), **optional dependency conflicts and workarounds**, and **all fixes and behavior changes**. It can be very long; use the table of contents to jump to a section.

---

## Table of contents

1. [Summary of modifications](#1-summary-of-modifications)
2. [How to run the application](#2-how-to-run-the-application)
3. [Text detection – all modules and how to run](#3-text-detection--all-modules-and-how-to-run)
4. [OCR – all modules and how to run](#4-ocr--all-modules-and-how-to-run)
5. [Inpainting – all modules and how to run](#5-inpainting--all-modules-and-how-to-run)
6. [Translation modules](#6-translation-modules)
7. [Settings reference: inpaint size, mask dilation, detection, OCR, formatting](#7-settings-reference-inpaint-size-mask-dilation-detection-ocr-formatting)
8. [Optional dependency conflicts and workarounds](#8-optional-dependency-conflicts-and-workarounds)
9. [New and modified files](#9-new-and-modified-files)
10. [Fixes and behavior changes](#10-fixes-and-behavior-changes)
11. [Documentation and references](#11-documentation-and-references)

---

## 1. Summary of modifications

This fork adds **many new optional modules** and applies **fixes and setting improvements**. Original behavior and defaults are unchanged unless noted. New modules are discovered automatically via the existing registry (no changes to core launch or config flow). You only install extra dependencies for the modules you use.

| Category | What was added / changed |
|----------|---------------------------|
| **Text detection** | MMOCR, PP-OCRv5, Surya, Magi (Manga Whisperer), TextMamba (stub), **CRAFT** (standalone), **HF object-detection** (default: ogkalu comic-text-and-bubble-detector), DPText-DETR, SwinTextSpotter v2 (optional repos). |
| **OCR** | 20+ new OCR backends: TrOCR, GOT-OCR2, GLM-OCR, Donut, PaddleOCR-VL (HF), Qwen2-VL 7B, DeepSeek-OCR, LightOn, Chandra, DocOwl2, Nanonets, Ocean-OCR, InternVL2/3, Florence-2, MiniCPM-o, OCRFlux, **HunyuanOCR**, **Manga OCR Mobile** (TFLite), **Nemotron Parse** (full-page). |
| **Inpainting** | Simple LaMa, Diffusers (SD 1.5, SD2 768, SDXL 1024, DreamShaper, FLUX Fill, Kandinsky), **RePaint**, **LaMa ONNX** (general + manga), **Qwen-Image-Edit**, **MAT** (repo+checkpoint), **CUHK Manga**, **Fluently v4**. |
| **Translation** | No new translators in this fork; existing LLM_API_Translator, Sakura, DeepL, etc. unchanged. |
| **Settings / fixes** | **Mask dilation** configurable (0–5) for lama_large_512px; **inpaint_size** options per inpainter; small-bubble normalization for lama_mpe; **crop_padding** for OCRs; CTD **box score threshold** and **merge tolerance**; **hf_object_det** default model_id = ogkalu/comic-text-and-bubble-detector; optional dependency docs (craft_det, simple_lama). |
| **Documentation** | `docs/BEST_MODELS_RESEARCH.md`, `docs/MODELS_REFERENCE.md`, `docs/QUALITY_RANKINGS.md` (tiered quality/accuracy), `docs/OPTIONAL_DEPENDENCIES.md`, `docs/INSTALL_EXTRA_DETECTORS.md`, `docs/MANHUA_BEST_SETTINGS.md`, this README. |

---

## 2. How to run the application

### Base setup (same as original)

- **Python:** 3.10 or 3.11 (≤ 3.12; avoid Microsoft Store Python).
- **Git:** Installed and in PATH.

```bash
# Clone the repository (or your fork)
git clone https://github.com/dmMaze/BallonsTranslator.git
cd BallonsTranslator

# First run: installs base deps and downloads default models into data/
python launch.py

# Update dependencies and code
python launch.py --update
```

If model downloads fail, use the original README links (MEGA / Google Drive) and place the `data` folder in the project root.

### Running

- **GUI:**  
  `python launch.py`  
  Then open the **settings panel** and choose **Text detection**, **OCR**, **Inpainting**, and **Translation** from the dropdowns. New modules appear automatically. Set **device** (CPU/CUDA) and any model-specific options there.

- **Headless (no GUI):**  
  `python launch.py --headless --exec_dirs "[DIR_1],[DIR_2]..."`  
  Settings are read from `config/config.json`. Ensure the chosen detector, OCR, and inpainter are installed and configured in that config.

- **Logical DPI (font/rendering):**  
  If rendered font size is wrong, use `--ldpi 96` (or 72) as needed:  
  `python launch.py --ldpi 96`

---

## 3. Text detection – all modules and how to run

Select the detector from the **Text detection** dropdown in the settings panel. Pair detection-only modules with an OCR (see [INSTALL_EXTRA_DETECTORS.md](docs/INSTALL_EXTRA_DETECTORS.md) for which work with **none_ocr**).

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
| **mmocr_det** | `pip install openmim` then `mim install mmengine mmcv mmdet mmocr` (see `doc/INSTALL_MMOCR.md` for Windows). | Select **mmocr_det**; pair with mmocr_ocr. |
| **paddle_det_v5** | `pip install paddlepaddle paddleocr` (3.x). | Select **paddle_det_v5**; pair with paddle_rec_v5. |
| **surya_det** | `pip install surya-ocr` | Select **surya_det**; pair with surya_ocr. |
| **magi_det** | `pip install transformers torch einops` | Select **magi_det**; model downloads from HF (ragavsachdeva/magi) on first use; pair with any OCR. |
| **craft_det** | `pip install craft-text-detector torch` | Select **craft_det**; outputs 4-point quads for merge; pair with any OCR. **Conflict:** needs opencv<4.5.4.62; see [Optional dependencies](#8-optional-dependency-conflicts-and-workarounds). |
| **dptext_detr** | Clone [DPText-DETR](https://github.com/ymy-k/DPText-DETR), install its deps. | Select **dptext_detr**; set **repo_path** to your clone; pair with any OCR. |
| **swintextspotter_v2** | Clone [SwinTextSpotterv2](https://github.com/mxin262/SwinTextSpotterv2), install its deps. | Select **swintextspotter_v2**; set **repo_path**; use **none_ocr** if demo outputs text. |
| **hunyuan_ocr_det** | Same as hunyuan_ocr (transformers, etc.). | Select **hunyuan_ocr_det**; set OCR to **none_ocr** to keep spotter text. |
| **textmamba_det** | None (stub) | Selecting it raises an error until official code is released; use mmocr_det or surya_det meanwhile. |

---

## 4. OCR – all modules and how to run

Select the OCR from the **OCR** dropdown. Install only the dependencies for the modules you use.

### Original / built-in

| Module | How to run |
|--------|-------------|
| **paddle_rec_v5**, **paddle_ocr** | Paddle stack; select and set language, device. |
| **PaddleOCRVLManga**, **paddle_vl** | VLM manga or server; select and configure. |
| **manga_ocr** | Select **manga_ocr**; model in `data/models/manga-ocr-base` (auto-download). |
| **easyocr_ocr**, **mmocr_ocr** | Pair with corresponding detector. |
| **mit32px**, **mit48px**, **mit48px_ctc** | From manga-image-translator; select as needed. |
| **google_vision**, **bing_ocr**, **one_ocr**, **windows_ocr**, **macos_ocr**, **llm_ocr**, **stariver_ocr**, **none_ocr** | Select and configure (API keys, etc.). **none_ocr** = no OCR (use with spotters). |

### New in this fork

| Module | Dependencies | How to run |
|--------|--------------|------------|
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

No new translators were added in this fork. Use the existing **LLM_API_Translator** (GPT-4o/Claude/Gemini), **ChatGPT**, **Sakura** (JP↔EN), **DeepL**, **google**, **nllb200**, **m2m100**, **Sugoi**, **t5_mt**, **opus_mt**, **Baidu**, **Youdao**, **Caiyun**, **Papago**, **Yandex**, **text-generation-webui**, **None**, **Copy Source**. Configure API keys and endpoints in the settings panel. See original README and `doc/加别的翻译器.md` for adding new translators.

---

## 7. Settings reference: inpaint size, mask dilation, detection, OCR, formatting

### 7.1 CTD (ComicTextDetector)

- **detect_size:** e.g. 1280 (higher = better quality, slower). Up to 2400 supported.
- **box score threshold:** 0.35–0.48 typical; lower = more boxes (e.g. 0.42–0.45 for manhua).
- **merge font size tolerance:** e.g. 3.0; higher = merge more lines into one bubble.
- **mask dilate size:** 2 default; kernel for mask dilation at detection stage.
- **min box area:** 0 or 100–200 to drop tiny noise.
- **custom_onnx_path:** Optional path to alternate ONNX (e.g. mayocream comic-text-detector); leave empty for default CTD ONNX.

### 7.2 Inpaint size (all inpainters)

- **lama_large_512px:** Options 512, 768, 1024, 1536, 2048. Default 1024. Smaller = less VRAM, gentler on small bubbles; larger = more detail on big regions. Avoid 2048 unless needed (risk of artifacts).
- **lama_mpe, aot:** 1024 or 2048.
- **Diffusers-based (SD, SD2, SDXL, DreamShaper, Fluently, Kandinsky, RePaint, Qwen-Image-Edit):** Each has an **inpaint_size** (or similar) in params; 512/768/1024 typical. Match to model native resolution when possible.
- **lama_onnx:** 512 (model is 512×512); param controls max side before resize.
- **lama_manga_onnx:** Often 1024 default; stride 64 for alignment.
- **mat:** 512 typical.

### 7.3 Mask dilation (kernel)

- **lama_large_512px** exposes **mask_dilation** (0–5). It sets **mask_dilation_iterations** for a 3×3 morphological dilation on the mask before inpainting. **0** = no dilation; **2** = default (balanced); **3–5** = more coverage for dots/smudges; **0–1** = minimal distortion on tiny bubbles.
- Base inpainter (`modules/inpaint/base.py`) applies this dilation in `inpaint()` so all block-based inpainters that inherit (e.g. lama_large_512px) use it. Other inpainters (AOT, lama_mpe, Diffusers, etc.) do not expose a separate mask_dilation param; the base applies a configurable **mask_dilation_iterations** only when the inpainter sets it (lama_large_512px).

### 7.4 OCR crop padding

Many OCRs have **crop_padding** (pixels to add around each detected box when cropping for OCR). Typical range 0–24. **6–8** is a good default to reduce clipped text at edges (e.g. with CTD). **manga_ocr**, **surya_ocr**, **trocr**, **manga_ocr_mobile**, etc. expose this in params.

### 7.5 Text and box formatting

- **Global font format:** In settings panel → **嵌字** (typesetting), the “global font format” is the format used when no text block is selected; you can set default font, size, color, alignment, etc.
- **Per-block formatting:** In text edit mode, select a block and use the right-hand font/format panel (bold, italic, underline, alignment, letter spacing, line spacing, vertical text). Supports rich text and presets.
- **Box/block layout:** Detection produces boxes (quadrilaterals); the app keeps **lines** (polygon points) per block. Merge/split and reading order depend on detector and post-processing (e.g. CTD merge tolerance, Paddle strict bubble mode). No changes to core box data format in this fork; new detectors return the same `(mask, List[TextBlock])` interface.

### 7.6 Paddle detection (strict bubble mode)

For **paddle_det**, **Strict bubble mode** applies stricter thresholds and filters (min_detection_area, max_aspect_ratio, box_shrink_px, merge_same_line_only, merge_line_overlap_ratio). Useful for comics so different bubbles are not merged. **det_limit_side_len** can be set (e.g. 960 when using Ocean OCR on CPU to avoid timeout).

---

## 8. Optional dependency conflicts and workarounds

Some optional modules require dependency versions that **conflict** with the main `requirements.txt`. See **docs/OPTIONAL_DEPENDENCIES.md** for full detail.

| Module | Conflict | Workaround |
|--------|----------|------------|
| **craft_det** | `craft-text-detector` needs **opencv-python<4.5.4.62**; main app uses **opencv≥4.8**. | Use **easyocr_det** or **mmocr_det** instead; or install in a separate venv with older opencv. If you keep main opencv, **craft_det** may not register (version check in code) or fail at runtime. |
| **simple_lama** | `simple-lama-inpainting` needs **pillow<10**; project uses **Pillow 10.x**. | Use **lama_large_512px**, **lama_onnx**, or **lama_manga_onnx** instead; or downgrade Pillow in a separate venv. |

The main application and all other modules work with the versions in `requirements.txt`.

---

## 9. New and modified files

### New files (no removals of original files)

- **Text detection:**  
  `modules/textdetector/detector_mmocr.py`, `detector_paddle_v5.py`, `detector_surya.py`, `detector_magi.py`, `detector_textmamba.py`, `detector_craft.py`, `detector_hf_object_detection.py`, `detector_dptext_detr.py`, `detector_swintextspotter_v2.py` (existing in original may differ; this fork adds or modifies as listed).

- **OCR:**  
  `modules/ocr/ocr_trocr.py`, `ocr_got_ocr2.py`, `ocr_glm_ocr.py`, `ocr_donut.py`, `ocr_paddleocr_vl_hf.py`, `ocr_qwen2vl.py`, `ocr_deepseek.py`, `ocr_lighton.py`, `ocr_chandra.py`, `ocr_docowl2.py`, `ocr_nanonets.py`, `ocr_ocean.py`, `ocr_internvl2.py`, `ocr_internvl3.py`, `ocr_florence2.py`, `ocr_minicpm.py`, `ocr_ocrflux.py`, `ocr_hunyuan.py`, `ocr_manga_mobile.py`, `ocr_nemotron.py`.

- **Inpainting:**  
  `modules/inpaint/inpaint_simple_lama.py`, `inpaint_diffusers_sd.py`, `inpaint_sd2.py`, `inpaint_sdxl.py`, `inpaint_dreamshaper.py`, `inpaint_flux_fill.py`, `inpaint_kandinsky.py`, `inpaint_fluently.py`, `inpaint_cuhk_manga.py`, `inpaint_repaint.py`, `inpaint_lama_onnx.py`, `inpaint_lama_manga_onnx.py`, `inpaint_qwen_image_edit.py`, `inpaint_mat.py`.

- **Documentation:**  
  `docs/BEST_MODELS_RESEARCH.md`, `docs/MODELS_REFERENCE.md`, `docs/QUALITY_RANKINGS.md`, `docs/OPTIONAL_DEPENDENCIES.md`, `docs/INSTALL_EXTRA_DETECTORS.md`, `docs/MANHUA_BEST_SETTINGS.md`, this file `README_MODIFICATIONS.md`.  
  `doc/INSTALL_MMOCR.md` (if present).

### Unchanged (behavior and discovery)

- `modules/base.py`: `MODULE_SCRIPTS` and module discovery unchanged; new modules are picked up by existing `ocr_*.py`, `inpaint_*.py`, `detector_*.py` patterns.
- `launch.py`, config flow, and UI flow unchanged; new options appear in the same dropdowns.

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

- **Config:**  
  - Defaults in `config/config.json` (e.g. ctd box score threshold, detect_size, inpainter choice) are unchanged unless you alter them; new modules appear when their dependencies are installed.

---

## 11. Documentation and references

| Document | Description |
|----------|-------------|
| **docs/QUALITY_RANKINGS.md** | Tier-based quality/accuracy rankings for all detection, OCR, and translation modules; task-based SOTA (document vs manga vs multilingual); sanity-check notes. |
| **docs/MODELS_REFERENCE.md** | Map of recommended models to BallonsTranslator modules; quick reference and “not integrated” list. |
| **docs/BEST_MODELS_RESEARCH.md** | Detailed research on OCR, detection, inpainting; benchmarks and recommendations. |
| **docs/OPTIONAL_DEPENDENCIES.md** | craft_det and simple_lama dependency conflicts and workarounds. |
| **docs/INSTALL_EXTRA_DETECTORS.md** | Optional detectors (SwinTextSpotter, DPText-DETR, CRAFT, hf_object_det); none_ocr usage; detection vs OCR coverage table. |
| **docs/MANHUA_BEST_SETTINGS.md** | Recommended detection, OCR, and inpainting settings for manhua (Chinese comics). |
| **Original README** | [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) – base setup, Windows/Mac, translators, AMD ROCm/ZLUDA. |
| **doc/加别的翻译器.md** | How to add new translators. |

---

If a module is missing from the dropdown or fails to load, check the console/log for import errors and install the dependencies listed in the tables above. For quality-focused choices, use **docs/QUALITY_RANKINGS.md** and **docs/MANHUA_BEST_SETTINGS.md**.
