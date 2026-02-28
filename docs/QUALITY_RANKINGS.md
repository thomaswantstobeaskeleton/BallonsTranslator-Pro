# Quality & Accuracy Rankings: Detection, OCR, Translation

**Criteria:** Quality and accuracy only. Speed is ignored. This document ranks every text detection, OCR, and translation model/implementation in BallonsTranslator. Where benchmarks show **performance clusters** rather than clear 1→N separation, **tier-based rankings** are used instead of strict linear order, so the list is easier to defend against published benchmarks and community consensus (see sanity-check notes below).

**Scope:** Detection → finding text regions (boxes/masks). OCR → recognizing text inside regions. Translation → converting source text to target language. Inpainting is out of scope here; see the [Diffusion vs non-diffusion](#4-diffusion-vs-non-diffusion-note) note for how diffusion-based inpainting compares for text removal.

---

## 1. Text Detection (Tiered by Quality / Accuracy)

**How to read:** Tier 1 = best; Tier 4 = baseline/special-case. Order within a tier is **not** a strict quality ranking. Benchmarks: CTW1500, TotalText, ICDAR, COMICS Text+, and manga/comic-specific use.

### Tier 1 — Manga / Comic Detection (SOTA for manga workflows)

Best for: speech bubbles, panels, vertical Japanese, comic layout. These are the defensible top choices for manga; ordering between them is debatable (CTD = ecosystem standard, ogkalu RT-DETR can match or exceed on raw box accuracy, Magi adds reading order and structure).

| Module | Model | Primary strength |
|--------|--------|-------------------|
| **ctd** | ComicTextDetector | De facto manga baseline; optimized for bubbles & vertical text; strong community validation. |
| **hf_object_det** | ogkalu/comic-text-and-bubble-detector (default) | RT-DETR-v2 fine-tuned on ~11k manga/webtoon/manhua/comic; transformer-based; strong box accuracy. |
| **magi_det** | Manga Whisperer (Magi, CVPR 2024) | Unified: text boxes, panels, characters, reading order; best when structure matters beyond raw boxes. |

### Tier 2 — Strong Scene / Document Detection (Benchmark-competitive)

Best for: general document and scene text. MMOCR (DBNet++) often leads ICDAR/TotalText-style benchmarks; Surya is strong for multilingual and document; Paddle v5 is competitive. Ordering here reflects benchmark strength and multilingual robustness.

| Module | Model | Primary strength |
|--------|--------|-------------------|
| **mmocr_det** | MMOCR (DBNet/DBNet++/FCENet/PSENet/TextSnake) | DBNet++ among top on ICDAR/TotalText; TextSnake for curved text. |
| **surya_det** | Surya detection | Line-level; 90+ languages; Segformer; strong multilingual and document. |
| **paddle_det_v5** | PP-OCRv5 detection | Handwriting, vertical, rotated, curved; multi-language. |
| **paddle_det** | PaddleOCR DB/DB++ | Mature document + scene; good fallback when CTD misses. |

### Tier 3 — General / Classical Detection

| Module | Model | Notes |
|--------|--------|--------|
| **craft_det** | CRAFT (craft-text-detector) | Curved and scene text; historically strong. Requires opencv<4.5.4.62 (see OPTIONAL_DEPENDENCIES.md). |
| **ysgyolo** | YOLO (e.g. ogkalu comic-speech-bubble-yolov8m) | Good for speech-bubble-only with comic-trained weights. |
| **dptext_detr** | DPText-DETR | DETR-based scene text; optional repo. |
| **swintextspotter_v2** | SwinTextSpotter v2 | Spotter (det+rec); optional repo; high quality, heavier setup. |
| **easyocr_det** | EasyOCR detection | CRAFT-based; decent general use; outclassed by Tier 2 for accuracy. |

### Tier 4 — Spotter / API / Stub

| Module | Model | Notes |
|--------|--------|--------|
| **hunyuan_ocr_det** | HunyuanOCR spotting | Full-image spotting (det+rec in one); use with none_ocr. |
| **stariver_ocr** | Stariver API | Boxes+text via API; use with none_ocr; quality = service. |
| **textmamba_det** | TextMamba (stub) | Not runnable; official code not released. Paper: CTW1500 89.7%, TotalText 89.2%. |

---

## 2. OCR (Tiered by Quality & Accuracy — Benchmark-Driven)

**How to read:** Tier 1 = state-of-the-art level; Tier 4 = baseline/legacy/platform. **Order within a tier is not a strict rank**—benchmarks (OmniDocBench, olmOCR-Bench, DocVQA, OCRBench) show clustering, not clear 1→N separation. Primary strength (document / scene / multilingual / manga) is called out so you can choose by task.

### Tier 1 — State-of-the-Art OCR

#### Tier 1A — Document parsing SOTA

Best for: structured documents, tables, formulas, charts, multi-layout parsing.

| Module | Model | Primary strength |
|--------|--------|-------------------|
| **paddleocr_vl_hf** | PaddleOCR-VL (HF) | OmniDocBench leader (~92.8); 109 languages; strong structured parsing. |
| **ocean_ocr** | Ocean-OCR 3B | Beats classic OCR in doc/scene tasks; strong edit-distance metrics. |
| **internvl2_ocr** | InternVL2 (8B/2B) | DocVQA ~91.6; strong OCRBench; robust layout reasoning. |
| **chandra_ocr** | Chandra 9B | ~83% olmOCR-Bench; strong layout & handwriting; 40+ languages. |
| **hunyuan_ocr** | HunyuanOCR-1B | SOTA in <3B class on OCRBench; ICDAR 2025 DIMT small-model track. |

#### Tier 1B — Strong multilingual / compact SOTA-class

Best for: high accuracy under size constraints.

| Module | Model | Primary strength |
|--------|--------|-------------------|
| **lighton_ocr** | LightOnOCR-2-1B | ~83% olmOCR; very strong per-parameter efficiency. |
| **got_ocr2** | GOT-OCR2 (580M) | High quality/size ratio; tables + formulas. |
| **internvl3_ocr** | InternVL3 family | Competitive VLM OCR across 1B/2B/8B. |

### Tier 2 — Strong All-Round OCR

#### Tier 2A — General VLM OCR extractors

| Module | Model | Notes |
|--------|--------|--------|
| **qwen2vl_7b** | Qwen2.5-VL 7B / OlmOCR 7B | Strong extraction; heavy VRAM. |
| **deepseek_ocr** | DeepSeek-OCR | ~75% olmOCR; good layout handling. |
| **nanonets_ocr** | Nanonets-OCR2-3B | Document/scene hybrid VLM. |
| **ocrflux** | OCRFlux 3B | Modern VLM-style document OCR. |
| **docowl2_ocr** | DocOwl2 9B | OCR-free document understanding; tables, layout. |
| **nemotron_ocr** | Nemotron Parse v1.1 | Full-page structured parsing; assign to blocks by overlap. |
| **glm_ocr** | GLM-OCR 0.9B | Lightweight document (text/formula/table); chat-style. |
| **minicpm_ocr** | MiniCPM-o | Compact VLM OCR. |
| **florence2_ocr** | Florence-2 | Microsoft vision; base/large; crop-based. |

#### Tier 2B — Strong multilingual OCR engines

| Module | Model | Notes |
|--------|--------|--------|
| **surya_ocr** | Surya | Excellent multilingual + handwriting; 90+ languages. |
| **paddle_rec_v5** | PP-OCRv5 recognition | Strong document OCR; pairs with paddle_det_v5. |
| **paddle_vl** | Paddle-VL (server) | High quality when server is tuned. |

#### Tier 2C — Manga-optimized OCR

For **clean Japanese manga speech bubbles**, these can outperform larger VLMs in CER despite lower general-tier placement.

| Module | Model | Notes |
|--------|--------|--------|
| **PaddleOCRVLManga** | PaddleOCR-VL Manga | Tuned for comic layout. |
| **manga_ocr** | Manga OCR (kha-white) | Very strong for Japanese speech bubbles. |

### Tier 3 — Solid / Mid-Tier OCR

| Module | Model | Notes |
|--------|--------|--------|
| **paddle_ocr** | PaddleOCR (legacy) | Mature baseline; outperformed by paddle_rec_v5 / PaddleOCR-VL. |
| **trocr** | TrOCR | Strong English printed/handwritten; not for CJK. |
| **donut** | Donut | Task-based DocVQA/CORD. |
| **one_ocr** | OneOCR | Vendor-based local DLL; platform-dependent. |
| **google_vision** | Google Cloud Vision API | Strong commercial baseline; can match or exceed many open models on clean docs. |
| **llm_ocr** | LLM API OCR | Quality = chosen model (e.g. GPT-4o/Gemini); variable vs structured benchmarks. |
| **stariver_ocr** | Stariver API | API-dependent. |
| **bing_ocr** | Bing OCR / Lens | Platform quality. |
| **easyocr_ocr** | EasyOCR | Good general-purpose; lower ceiling than VLM/specialized. |
| **mmocr_ocr** | MMOCR (SAR/CRNN) | Classical pipeline; pairs with mmocr_det. |

### Tier 4 — Lightweight / Platform / Legacy

| Module | Model | Notes |
|--------|--------|--------|
| **manga_ocr_mobile** | Manga OCR Mobile (TFLite) | Lightweight; ~7.4% CER Manga109s; worse on English/punctuation. |
| **mit32px** / **mit48px** / **mit48px_ctc** | MIT 32/48px | Legacy; limited vs modern models. |
| **windows_ocr** | Windows OCR | Platform engine. |
| **macos_ocr** | macOS OCR | Platform engine. |
| **google_lens_exp** | Google Lens (experimental) | Experimental; quality varies. |
| **none_ocr** | None | No OCR; for use with spotters that output text (e.g. hunyuan_ocr_det, stariver). |

### Task-based OCR SOTA summary

| Task | Recommended modules |
|------|---------------------|
| **Best document OCR (structured parsing)** | paddleocr_vl_hf, ocean_ocr, internvl2_ocr, chandra_ocr |
| **Best compact (<3B) high-accuracy** | hunyuan_ocr, lighton_ocr, got_ocr2 |
| **Best multilingual general OCR** | surya_ocr, paddle_rec_v5, InternVL family |
| **Best manga speech-bubble OCR** | manga_ocr, PaddleOCRVLManga |

---

## 3. Translation (Tiered by Quality / Accuracy)

**How to read:** Tier 1 = best contextual/quality; Tier 4 = no or copy-only. Quality is **language-pair and domain dependent**. LLM_API_Translator and ChatGPT both use the same model family (e.g. GPT-4o) when provider is OpenAI; they are grouped to avoid redundant ranking.

### Tier 1 — Best contextual translation (LLM / API)

| Module | Model | Notes |
|--------|--------|--------|
| **LLM_API_Translator** | GPT-4o / Claude / Gemini (OpenAI/OpenRouter) | Best nuance, register, terminology; use with appropriate provider. |
| **ChatGPT** / **ChatGPT_exp** | ChatGPT API | Same model family as LLM_API (OpenAI); strong quality. |
| **Sakura** | Sakura (JP↔EN) | Specialized for Japanese↔English; often preferred for manga/anime dialogue (domain-dependent). |

### Tier 2 — Strong commercial / local MT

| Module | Model | Notes |
|--------|--------|--------|
| **DeepL** / **DeepLX API** | DeepL | Excellent for many pairs including Japanese; often best among non-LLM APIs. |
| **DeepL Free** | DeepL free tier | Same engine as DeepL with limits. |
| **google** | Google Translate API | Strong coverage and quality; good default API. |
| **nllb200** | NLLB-200 (local) | 200 languages; CTranslate2; good local multilingual. |
| **m2m100** | M2M-100 (local) | Many-to-many; local CTranslate2. |
| **Sugoi** | Sugoi | Japanese-focused local option. |

### Tier 3 — Good / flexible translation

| Module | Model | Notes |
|--------|--------|--------|
| **t5_mt** | T5 MT (prompt-based) | Quality depends on model and prompt. |
| **text-generation-webui** | Local LLM (oobabooga etc.) | Quality = chosen model. |
| **opus_mt** | OPUS-MT (Helsinki-NLP) | Per-pair; good for supported pairs. |
| **Baidu** / **Youdao** / **Caiyun** | Baidu / Youdao / Caiyun APIs | Solid for Chinese-centric pairs. |
| **Papago** | Papago | Good for Korean. |
| **Yandex** / **Yandex-FOSWLY** | Yandex | Good coverage; quality varies by pair. |
| **ezTrans** | ezTrans | Local; often used for Korean. |
| **TranslatorsPack** | Translators (library) | Aggregates backends; quality = selected backend. |

### Tier 4 — No translation

| Module | Notes |
|--------|--------|
| **None** | Pass-through. |
| **Copy Source** | Copy text as-is. |

---

## 4. Diffusion vs Non-Diffusion Note

For **detection and OCR** there is no “diffusion” vs “non-diffusion” split: detectors and OCRs are discriminative or encoder-decoder VLMs, not diffusion generative models.

**Inpainting (text removal):**

- **Non-diffusion (e.g. LaMa, AOT, MAT):** For **text removal in manga**, **lama_large_512px** (dreMaz/AnimeMangaInpainting) is generally **more accurate and stable** than diffusion: fewer squiggly or text-like artifacts, no hallucinated strokes.
- **Diffusion-based (SD inpainting, FLUX Fill, DreamShaper, etc.):** Can produce high-fidelity texture but often **less accurate for strict text removal**: faint remnants or plausible-but-wrong strokes. Benchmarks (e.g. FID) do not always reflect “perfect text erasure.”

**Recommendation:** For quality/accuracy of text removal, prefer **lama_large_512px** or **lama_manga_onnx** over diffusion-based inpainters.

---

## 5. Sources and Benchmarks

- **OCR:** OmniDocBench (CVPR 2025), olmOCR-Bench, DocVQA, OCRBench; PaddleOCR-VL 92.86 OmniDocBench; Chandra 83.1% olmOCR; Ocean-OCR beating Paddle/TextIn; InternVL2 DocVQA 91.6, OCRBench 794; HunyuanOCR SOTA <3B OCRBench / ICDAR 2025 DIMT; LightOn 83.2% OlmOCR.
- **Detection:** CTW1500, TotalText, ICDAR; DBNet++/MMOCR leaderboards; Magi CVPR 2024; ComicTextDetector community use; ogkalu RT-DETR comic fine-tune; TextMamba (stub) paper metrics.
- **Translation:** WMT metrics (BLEU/COMET); LLM-based translation; DeepL/GPT-4/Claude comparisons; Sakura for JP↔EN manga domain.
- **Inpainting:** LaMa vs diffusion for text removal (BEST_MODELS_RESEARCH.md, project docs).

---

## 6. How to Use This List

- **Detection:** For manga use **ctd** or **hf_object_det** (ogkalu) or **magi_det**; for general/scene text use **mmocr_det** or **surya_det** or **paddle_det_v5**.
- **OCR:** Use **Task-based OCR SOTA** (Section 2): document parsing → Tier 1A; manga bubbles → Tier 2C; multilingual → Tier 2B; compact → Tier 1B.
- **Translation:** Prefer **LLM_API_Translator** or **Sakura** (JP↔EN) for best quality; **DeepL** or **google** for API; **nllb200** for local multilingual.

Rankings are indicative and depend on language pair, domain, and input quality; re-run evaluations on your own data when possible.

---

## 7. Sanity-Check and Methodology Note

This document was revised to use **tier-based rankings** where benchmarks and community consensus do not support a strict linear 1→N order:

- **Detection:** Top tier (CTD, ogkalu, Magi) is defensible for manga; MMOCR vs Surya vs Paddle ordering was relaxed into Tier 2 with benchmark-based rationale.
- **OCR:** Strict linear ordering was replaced with tiers (Tier 1–4) and task-based SOTA (document / compact / multilingual / manga), so the list does not overstate separation between near-equal models (e.g. PaddleOCR-VL, Ocean-OCR, InternVL2, Chandra).
- **Translation:** LLM_API and ChatGPT were grouped to avoid redundant ranking; Sakura placement is noted as domain-dependent (JP↔EN manga).
- **Diffusion note:** Unchanged; aligns with empirical manga pipelines and artifact behavior in diffusion inpainting.

This keeps the list **benchmark-consistent and academically defensible** while still listing every module.
