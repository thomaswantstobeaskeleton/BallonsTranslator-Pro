# Best Modules and Settings Guide

This document summarizes the **recommended options** for text detection, OCR, inpainting, and translation in BallonsTranslator, plus **recommended settings** and notes on **AI models** (including what you have and optional upgrades).

---

## 1. Text detection

**Available modules:** `ctd`, `paddle_det`, `easyocr_det`, `ysgyolo`, `stariver_ocr`

### Best choice: **ctd (ComicTextDetector)**

- **Why:** Same detector as [manga-image-translator](https://github.com/zyddnys/manga-image-translator); tuned for manga/comic speech bubbles and text regions. Good balance of speed and accuracy; works on CPU (ONNX) or GPU (PyTorch).
- **Alternatives:**  
  - **paddle_det** – PaddleOCR detection only (DB/DB++). Strong for Chinese/document text; use when CTD misses regions or for non-comic layouts. Requires `paddleocr`.  
  - **easyocr_det** – EasyOCR detector only (CRAFT or DBNet18). Good for Chinese (ch_sim), English, ja, ko; use when CTD or paddle_det miss text. Requires `pip install easyocr`; set language to **ch_sim+en** for manhua.  
  - **ysgyolo** – Use if you have a custom YOLO checkpoint for your comic style.  
  - **stariver_ocr** – Cloud-based (星河云/团子); use only if you rely on that service.

### Optional: **paddle_det** (PaddleOCR detection only)

Use when CTD misses text or for document-style pages. Same detection backbone as PaddleOCR (DB/DB++). Set **language** to **ch** for Chinese, **en** for English. Requires `paddleocr` (and optionally `paddlepaddle-gpu`).

### Optional: **easyocr_det** (EasyOCR detection only)

Uses CRAFT or DBNet18 detector; good for Chinese (ch_sim), English, Japanese, Korean. Set **language** to **ch_sim+en** for manhua. **detect_network**: **craft** (default) or **dbnet18**. Requires `pip install easyocr`. Only the detection model is loaded (recognizer=False).

### Recommended settings (ctd)

| Parameter | Recommended | Notes |
|-----------|-------------|--------|
| **detect_size** | **1280** | Higher = better accuracy on large pages, slower. Use 1024 for speed, 1280 for quality. |
| **det_rearrange_max_batches** | **4** (or 8 if you have VRAM) | Larger = faster on GPU, more VRAM. |
| **device** | **cuda** (if available) | Use **cpu** for no GPU; then ONNX is used. |
| **mask dilate size** | **2** | Slight dilation of the mask; increase if text edges are cut off. |
| **font size multiplier** | **1.0** | Adjust if detected font sizes are consistently off. |
| **font size max / min** | **-1** | Disabled by default; set positive values to clamp font size. |

---

## 2. OCR (text recognition)

**Available modules:** `mit32px`, `mit48px`, `mit48px_ctc`, `manga_ocr`, `PaddleOCRVLManga`, `paddle_ocr`, `paddle_vl`, `surya_ocr`, `google_vision`, `bing_ocr`, `one_ocr`, `llm_ocr`, `windows_ocr`, `macos_ocr`, `stariver_ocr`, `none_ocr`, etc.

### Best choices by use case

| Use case | Recommended | Reason |
|----------|-------------|--------|
| **Manga (Japanese)** | **mit48px** or **manga_ocr** | mit48px: best all-round (from manga-image-translator). manga_ocr: [manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base), very strong on Japanese manga text. |
| **Manhwa / Korean** | **mit48px** or **paddle_ocr** (lang: korean) | MIT supports multi-script; Paddle has dedicated Korean. |
| **Manhua / Chinese** | **mit48px**, **paddle_ocr** (lang: ch), or **surya_ocr** | Paddle and MIT are strong for Chinese; Surya adds 90+ languages and good mixed-script (zh+en). |
| **Highest quality (manga, any script)** | **PaddleOCRVLManga** | Vision–language model; best accuracy, needs more VRAM and disk (~2GB model). You already have it under `data/models/PaddleOCR-VL-For-Manga`. |
| **Multilingual (zh, en, ja, ko, etc.)** | **surya_ocr** | 90+ languages; set language to "Chinese + English" or "Multilingual". Requires `pip install surya-ocr` (Python 3.10+, PyTorch). |
| **No GPU / lightweight** | **mit32px** | Smaller/faster; slightly less accurate than mit48px. |
| **CTC (alternative decoder)** | **mit48px_ctc** | Try if mit48px gives misreads on certain fonts. |

### Recommended settings (MIT OCR: mit32px / mit48px / mit48px_ctc)

| Parameter | Recommended | Notes |
|-----------|-------------|--------|
| **chunk_size** | **16** (or **24** for long lines) | Batch size for line recognition. 16 is a good default; 24 can help with very long text lines. |
| **device** | **cuda** (if available) | OCR benefits from GPU. |

### Recommended settings (PaddleOCRVLManga)

| Parameter | Recommended | Notes |
|-----------|-------------|--------|
| **max_new_tokens** | **512** | Usually enough for one text block. |
| **device** | **cuda** | Model is large; CPU is slow. |

### Recommended settings (manga_ocr)

| Parameter | Recommended | Notes |
|-----------|-------------|--------|
| **device** | **cuda** (if available) | Speeds up inference. |

---

## 3. Inpainting (text removal / background fill)

**Available modules:** `lama_large_512px`, `lama_mpe`, `aot`, `patchmatch`, `opencv-tela`

### Best choice: **lama_large_512px**

- **Why:** [AnimeMangaInpainting](https://huggingface.co/dreMaz/AnimeMangaInpainting) LAMA variant; best quality for manga/comic inpainting. Supports **bf16** on CUDA (saves VRAM) and flexible **inpaint_size**.
- **Alternatives:**  
  - **lama_mpe** – LAMA MPE; good quality, smaller than large.  
  - **aot** – Older manga-image-translator inpainter; faster, lower quality.  
  - **patchmatch** / **opencv-tela** – No neural net; use only as fallback (e.g. low VRAM).

### Recommended settings (lama_large_512px)

| Parameter | Recommended | Notes |
|-----------|-------------|--------|
| **inpaint_size** | **1536** or **1024** | 1536: best quality/speed tradeoff. 1024: less VRAM, faster. 2048: highest quality, most VRAM. |
| **precision** | **bf16** (CUDA) / **fp32** (CPU) | Use bf16 on supported GPUs to reduce VRAM. |
| **device** | **cuda** (if available) | Inpainting is heavy; CPU is slow. |

### Recommended settings (lama_mpe / aot)

| Parameter | Recommended | Notes |
|-----------|-------------|--------|
| **inpaint_size** | **2048** (lama_mpe) or **2048** (aot) | Larger = better quality, more VRAM. |
| **device** | **cuda** | Same as above. |

---

## 4. Translation

**Available modules:** `google`, `DeepL`, `DeepL Free`, `Sakura`, `Sugoi`, `ChatGPT`, `Baidu`, `Caiyun`, `Papago`, `Youdao`, `Yandex`, `m2m100`, `LLM_API_Translator`, `text-generation-webui`, `ezTrans`, `Copy Source`, `None`, etc.

### Best choices by use case

| Use case | Recommended | Reason |
|----------|-------------|--------|
| **General / free / no API** | **google** (default) | Free, good quality, many languages. |
| **Japanese → English (quality)** | **Sakura** or **Sugoi** | Sakura: LLM-based, best nuance (needs API/key). Sugoi: offline, CTranslate2, very good for manga. |
| **Japanese → English (offline)** | **Sugoi** | No API; model in `data/models/sugoi_translator/` (must be installed manually). |
| **Professional / EU languages** | **DeepL** (paid) or **DeepL Free** | Often better than Google for European languages. |
| **Chinese** | **Baidu**, **Caiyun**, or **Youdao** | Good for 中文; may need API keys. |
| **Korean** | **Papago** or **Google** | Papago strong for Korean; Google is a solid fallback. |
| **No translation** | **Copy Source** or **None** | Use when you only want detection + OCR + inpainting. |

### Notes

- **Sakura:** Supports glossary; best when you configure source/target and optional dictionary.
- **Sugoi:** Only Japanese → English; no API, but requires manual download of the Sugoi CTranslate2 model to `data/models/sugoi_translator/` (see README or project issues).
- **Google:** Set **translate_source** and **translate_target** in Config → module (e.g. 日本語 → 简体中文).

---

## 5. Default config summary (best overall)

In **Config → Run / module** you can set:

| Module | Recommended value |
|--------|--------------------|
| **textdetector** | `ctd` |
| **ocr** | `mit48px` (or `manga_ocr` for Japanese-only; `PaddleOCRVLManga` for max quality) |
| **inpainter** | `lama_large_512px` |
| **translator** | `google` (or `Sakura` / `Sugoi` for Japanese→English) |

Then in each module’s **params** in the config panel:

- **ctd:** detect_size **1280**, device **cuda**, det_rearrange_max_batches **4**, mask dilate **2**.
- **mit48px:** chunk_size **16**, device **cuda**.
- **lama_large_512px:** inpaint_size **1536**, precision **bf16**, device **cuda**.

---

## 6. AI models on disk and optional upgrades

### What you already have (under `data/models/`)

| Path | Purpose |
|------|--------|
| `comictextdetector.pt` / `comictextdetector.pt.onnx` | CTD text detection (GPU/CPU). |
| `ocr_ar_48px.ckpt` | MIT 48px OCR (attention). |
| `mit48pxctc_ocr.ckpt` | MIT 48px CTC OCR. |
| `mit32px_ocr.ckpt` | MIT 32px OCR. |
| `lama_large_512px.ckpt` | LAMA large inpainting (best quality). |
| `lama_mpe.ckpt` | LAMA MPE inpainting. |
| `aot_inpainter.ckpt` | AOT inpainting. |
| `manga-ocr-base/` | manga_ocr (Japanese manga OCR). |
| `PaddleOCR-VL-For-Manga/` | PaddleOCR-VL manga OCR (high quality). |
| `pkuseg/` | Used by some translators (e.g. Chinese). |

These are the same or equivalent to the models used by manga-image-translator (beta-0.3) and the linked Hugging Face repos; no need to replace them unless a newer official release appears.

### Optional “better” or alternative models

- **Text detection:** The repo currently uses manga-image-translator beta-0.3 CTD. There is no drop-in replacement in this project yet. **ysgyolo** can be “better” only if you train or obtain a YOLO model suited to your comic style.
- **OCR:** You already have the best options:
  - **mit48px** (and mit48px_ctc, mit32px) from manga-image-translator.
  - **manga_ocr** (manga-ocr-base) and **PaddleOCRVLManga** for highest quality.
- **Inpainting:** **lama_large_512px** is already the recommended best; no upgrade needed unless a newer AnimeMangaInpainting or LAMA variant is released and integrated.
- **Translation:** For Japanese→English, **Sugoi** is the main “better” offline option; it requires manually placing the CTranslate2 model into `data/models/sugoi_translator/` (see community docs or issues for download links and folder layout).

If you add new model files (e.g. a new OCR or detector), place them in the paths expected by the corresponding module (see each module’s `download_file_list` or docstrings in `modules/`).

---

## 7. Quick reference: one-line “best” setup

- **Detection:** ctd, detect_size 1280, GPU.  
- **OCR:** mit48px (or manga_ocr for JP, PaddleOCRVLManga for max quality), chunk_size 16, GPU.  
- **Inpaint:** lama_large_512px, inpaint_size 1536, bf16, GPU.  
- **Translate:** google (or Sakura/Sugoi for JP→EN).

Adjust **device** to **cpu** for any module if you have no GPU; expect slower runs, especially for OCR and inpainting.

---

## 8. Manhua (Chinese comic) – e.g. Rebirth of the Urban Immortal Cultivator

For **manhua** (Chinese webcomic / comic), use these settings so detection, OCR, translation, and typesetting are optimal.

### Module choices

| Module | Recommended | Notes |
|--------|-------------|--------|
| **Text detector** | **ctd** | Same as manga; works well for manhua speech bubbles and panels. |
| **OCR** | **mit48px** or **PaddleOCRVLManga** | Both handle Chinese well. PaddleOCRVLManga is best quality; mit48px is faster. |
| **Inpainter** | **lama_large_512px** | Best for comic-style inpainting. |
| **Translator** | **google** or **LLM_API_Translator** (e.g. LM Studio) | Chinese → English: Google is free; LLM gives more natural dialogue if you run a local model. |

### Language (Config → Run / module)

- **Translate source:** **简体中文** (Simplified Chinese).  
- **Translate target:** **English** (or your target language).

These must match your raw (Chinese) and desired output (e.g. English).

### Typesetting (Config → Typesetting / global format, or per-block)

| Setting | Recommended | Why |
|---------|-------------|-----|
| **Font** | **Komika Text**, **Patrick Hand**, **ZCOOL KuaiLe**, or **Indie Flower** | Comic-style, readable for dialogue. Install extra fonts in `BallonsTranslator/fonts` if needed. |
| **Alignment** | **Center** | Fits speech bubbles; use Left/Right only for side captions. |
| **Line spacing** | **1.2** | Default; good for multi-line bubbles. |
| **Letter spacing** | **1.15** | Slight spread for readability. |
| **Vertical** | **Off** for English output | Manhua raws are often vertical Chinese; translation is usually horizontal. Leave **vertical** on only for blocks you want to keep vertical (e.g. some SFX). |
| **Stroke width** | **0** or small (e.g. 0.5–1 px) | Light outline can help text stand out on busy art; 0 is clean. |
| **Font size** | **24** (default) or match bubble | Adjust per block if detection over/under-estimates. |

### Merge mode (text blocks)

Manhua often has **vertical** Chinese in columns. When merging detected regions:

- Use **垂直合并** (vertical merge) or **先垂直后水平** if bubbles are stacked top-to-bottom.
- Default **最大垂直间隙** (~10 px) is usually fine; increase if adjacent bubbles are merged by mistake.

### Quick checklist for manhua (CN → EN)

1. **Config → Run:** Source **简体中文**, Target **English**.  
2. **Config → Typesetting:** Font = comic-friendly (e.g. Komika Text), Alignment = Center, vertical = Off for translated text.  
3. **Detection:** ctd, detect_size 1280.  
4. **OCR:** mit48px or PaddleOCRVLManga.  
5. **Inpaint:** lama_large_512px, inpaint_size 1536.  
6. **Translate:** Google or LLM (e.g. LM Studio with a small model).  
7. After run, tweak any block’s font/size/alignment in the text panel if needed.
