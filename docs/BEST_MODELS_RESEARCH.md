# Best Models Research: OCR, Text Detection & Inpainting

Deep research summary for state-of-the-art and recommended models for manga/comic translation (as of 2024–2025). This project already implements many of these; recommendations are marked. For a **single best-to-worst ranking by quality/accuracy** (all detectors, OCRs, translators), see **[docs/QUALITY_RANKINGS.md](QUALITY_RANKINGS.md)**.

---

## 1. OCR (Text Recognition)

### Benchmarks & Leaders (2024–2025)

| Model | Strengths | Languages | Notes |
|-------|-----------|-----------|--------|
| **PaddleOCR-VL 0.9B** | SOTA document parsing (92.86 OmniDocBench), 109 languages, tables/formulas | 109 | **Added** `paddleocr_vl_hf` (HF transformers); also PaddleOCRVLManga in project. |
| **Ocean-OCR (3B MLLM)** | Beats PaddleOCR/TextIn on doc + scene + handwritten | Multilingual | **Added** `ocean_ocr`; guoxy25/Ocean-OCR, HF, trust_remote_code; quality over speed. |
| **InternVL2 (8B/2B)** | DocVQA 91.6, OCRBench 794, chart/scene text | Multilingual | **Added** `internvl2_ocr`; OpenGVLab/InternVL2-8B or 2B, HF, trust_remote_code. |
| **Surya** | 90+ languages, layout/reading order, strong on mixed & handwritten | 90+ | **Already in project.** Best open-source all-rounder; GPL-3.0. |
| **PaddleOCR (PP-OCRv4)** | Strong document OCR, Apache 2.0, lightweight options | 80+ | **Already in project** (Paddle OCR, Paddle VL server, PaddleOCRVLManga). |
| **Manga OCR** | Tuned for manga/comic in-page text | Japanese | **Already in project.** Good for speech bubbles. |
| **TrOCR** (Microsoft) | Strong printed/handwritten English, HuggingFace | EN (printed/handwritten) | **Added.** Useful for Latin text in comics. |
| **GOT-OCR2** | Unified OCR (plain/scene/formatted), 580M, HuggingFace | Multilingual | **Added (full).** Per-block + batch, bf16/fp16, stop_strings; quality focus. |
| **GLM-OCR** | Lightweight 0.9B document OCR (text/formula/table), HuggingFace | 100+ | **Added.** Chat-style; good for documents. |
| **Donut** | OCR-free document understanding (DocVQA/CORD), HuggingFace | Task-based | **Added.** DocVQA = read text; CORD = receipt/form. |
| **OneOCR** | High accuracy, line/word-level, local DLL | Multilingual | **Already in project.** Good Windows/local option. |
| **GPT-4o / Gemini 2.5 Pro** | Top on some benchmarks (e.g. Chinese); API cost | Multilingual | Use via LLM OCR module if needed. |
| **Qwen2.5-VL 7B / OlmOCR 7B** | Heavyweight VLM OCR; Qwen general “extract text”, OlmOCR document-tuned (Allen AI) | Multilingual | **Added** `qwen2vl_7b`; quality over speed; ~16GB+ VRAM bf16. |
| **DeepSeek-OCR** | Document OCR, layout/tables, 75.7 OlmOCR-Bench; trust_remote_code | Multilingual | **Added** `deepseek_ocr`; heavyweight; prompt "<image>\\nFree OCR. ". |
| **LightOnOCR-2-1B** | 83.2% OlmOCR-Bench, 1B params; 3.3× faster than Chandra | Multilingual | **Added** `lighton_ocr`; quality/speed balance; pipeline or processor+model. |
| **Chandra 9B** | Top olmOCR (83.1%); layout, tables, math, 40+ languages | 40+ | **Added** `chandra_ocr`; chandra-ocr package; quality over speed. |
| **DocOwl2 9B** | OCR-free document understanding; tables, layout, multi-page | Multilingual | **Added** `docowl2_ocr`; mPLUG/DocOwl2, trust_remote_code. |
| **Nanonets-OCR2-3B** | 3B VLM OCR; document/scene text via chat | Multilingual | **Added** `nanonets_ocr`; HF transformers; messages + image path or PIL; quality-focused. |

### Recommendations for This Project

- **Manga/comic (CJK):** **Surya OCR** or **Manga OCR** for bubbles; **PaddleOCRVLManga** for VLM-based manga parsing.
- **Multilingual / document:** **Surya OCR** (default) or **Paddle OCR**.
- **English-only / Latin in manga:** **TrOCR** (printed or handwritten).
- **Local VLM, no server:** **PaddleOCRVLManga** (HuggingFace model).
- **Cloud / API:** **OneOCR**, **Google Vision**, or **LLM OCR** (GPT-4o/Gemini).
- **Unified HF OCR (quality):** **GOT-OCR2** (`got_ocr2`) for documents/comics when you want one strong model; heavier than TrOCR.
- **Lightweight HF document OCR:** **GLM-OCR** (`glm_ocr`, 0.9B) for text/formula/table; chat-style API.
- **OCR-free document:** **Donut** (`donut`) with DocVQA (“What is the text?”) or CORD for receipts/forms.
- **SOTA 109-language (HF):** **PaddleOCR-VL** (`paddleocr_vl_hf`) via transformers; prompt “OCR:” for text; requires transformers 5.x.
- **Heavyweight VLM OCR (quality):** **Qwen2.5-VL 7B** or **OlmOCR 7B** (`qwen2vl_7b`); use when quality matters more than speed; requires ~16GB+ VRAM (bf16).
- **Heavyweight document OCR:** **DeepSeek-OCR** (`deepseek_ocr`); layout/tables, multilingual; requires `trust_remote_code` and flash-attn optional.
- **Quality/speed balance OCR:** **LightOnOCR-2-1B** (`lighton_ocr`); 1B params, strong benchmarks; less VRAM than 7B VLMs.
- **Top benchmark (9B):** **Chandra** (`chandra_ocr`); install `chandra-ocr`; layout/tables/math; best olmOCR score among open models.
- **OCR-free document 9B:** **DocOwl2** (`docowl2_ocr`); mPLUG/DocOwl2 via transformers; tables, layout; trust_remote_code.
- **Heavyweight 3B VLM OCR:** **Nanonets-OCR2-3B** (`nanonets_ocr`); HF transformers; chat-style with image; quality over speed.
- **SOTA 3B OCR MLLM:** **Ocean-OCR** (`ocean_ocr`); first MLLM to beat dedicated OCR; doc/scene/handwritten; guoxy25/Ocean-OCR.
- **Document/chart/OCR 8B:** **InternVL2** (`internvl2_ocr`); OCRBench 794, DocVQA 91.6; OpenGVLab/InternVL2-8B or 2B.

---

## 2. Text Detection

### Benchmarks & Leaders (2024)

| Model | Use case | Notes |
|-------|----------|--------|
| **ComicTextDetector (CTD)** | Manga/comic bubbles and panels | **Already in project.** Primary detector for comics. |
| **Paddle OCR Det (DB/DB++)** | General document + scene text, Chinese/English | **Already in project** (`paddle_det`). Good when CTD misses text. |
| **TextMamba** | Scene text (CTW1500, TotalText, ICDAR) | Mamba-based; not yet integrated. |
| **CRAFT** | Curved text, scene text | **Added** `craft_det`; standalone CRAFT via craft-text-detector (pip). Use with any OCR. |
| **DBNet / DBNet++** | Document detection, scene text | **Added** `mmocr_det` (MMOCR); DBNet/DBNetpp/FCENet/PSENet; requires mmocr stack. |
| **PP-OCRv5 det** | Handwriting, vertical, rotated, curved; multi-language | **Added** `paddle_det_v5`; detection-only, PP-OCRv5_mobile_det/server_det (paddleocr 3.x). |
| **Surya det** | Line-level, 90+ languages, Segformer | **Added** `surya_det`; surya-ocr package; quality-focused document/comic. |
| **Manga Whisperer (Magi)** | Manga: panels, text boxes, characters, reading order | **Added** `magi_det`; ragavsachdeva/magi (v1), HF, trust_remote_code; CVPR 2024. |
| **TextMamba** | Scene text (CTW1500, TotalText, ICDAR); Mamba SSM | **Placeholder** `textmamba_det`; official code TBD (arXiv:2512.06657). |

### Recommendations for This Project

- **Primary (manga):** **CTD** with tuned merge tolerance and box score threshold (already improved).
- **Fallback / non-bubble:** **Paddle det** (`paddle_det`) with lower box thresh to catch more small text.
- **Two-step:** Run **CTD** first; optionally run **Paddle det** on remaining regions to fill misses.
- **Latest Paddle detection:** **PP-OCRv5** (`paddle_det_v5`) when using paddleocr 3.x; mobile or server det model, detection-only.
- **MMOCR (quality):** **mmocr_det** for DBNet/DBNetpp scene/document detection when mmengine, mmcv, mmdet, mmocr are installed.
- **Surya detection (line-level):** **surya_det** for 90+ language line detection when `surya-ocr` is installed; good for document/comic.
- **Manga (panels + text + reading order):** **magi_det** (Manga Whisperer) for manga/comic; panels, text boxes, characters; use any OCR after.
- **TextMamba:** **textmamba_det** is a placeholder until official code is released; use mmocr_det or surya_det for scene text meanwhile.

---

## 3. Inpainting (Text Removal)

### Benchmarks & Leaders (2022–2024)

| Model | Strengths | Notes |
|-------|-----------|--------|
| **LaMa** | Resolution-robust, large masks, Fourier convs | **Already in project** (lama_mpe, lama_large_512px). |
| **MAT (Mask-Aware Transformer)** | Large holes, high fidelity, CVPR 2022 | SOTA on Places/CelebA; 512×512; StyleGAN2-ADA base; complex to integrate. |
| **AOT (Generative Inpainting)** | Manga-image-translator default | **Already in project.** |
| **PatchMatch** | Fast, no GPU | **Already in project.** Good CPU fallback. |
| **RePaint (DDPM)** | Diffusion inpainting, arbitrary masks | **Added** `repaint`; Diffusers RePaintPipeline (e.g. google/ddpm-ema-celebahq-256); 256×256, optional. |
| **OpenCV LaMa ONNX** | General LaMa 512×512, ONNX | **Added** `lama_onnx`; opencv/inpainting_lama; set model_path. |
| **Simple LaMa** | Easiest LaMa API (pip) | **Added as optional.** Alternative LaMa backend. |
| **Stable Diffusion Inpaint (Diffusers)** | HF Diffusers, prompt-based | **Added as optional.** `diffusers_sd_inpaint`; slower, natural-looking fill. |
| **FLUX.1-Fill** | 12B rectified flow, inpainting/outpainting | **Added as optional.** `flux_fill`; high quality; use CPU offload if VRAM limited. |
| **Kandinsky 2.1 Inpaint** | CLIP + diffusion prior, Diffusers | **Added as optional.** `kandinsky_inpaint`; alternative to SD inpainting. |
| **SDXL Inpaint 1024** | 1024×1024 SDXL inpainting, quality-focused | **Added as optional.** `diffusers_sdxl_inpaint`; heavier/slower than SD 512; strength 0.99, more steps. |
| **DreamShaper Inpaint** | Lykon DreamShaper 8 inpainting, 512 | **Added as optional.** `dreamshaper_inpaint`; quality-focused SD-style inpainting. |
| **SD2 Inpaint 768** | stabilityai/stable-diffusion-2-inpainting, 768 | **Added as optional.** `diffusers_sd2_inpaint`; quality option between SD 1.5 and SDXL. |
| **Fluently v4 Inpaint** | fluently/Fluently-v4-inpainting (SD 1.5) | **Added as optional.** `fluently_v4_inpaint`; anime/comic style; small parts and complex objects. |
| **CUHK Manga Inpainting** | Seamless Manga Inpainting (SIGGRAPH 2021) | **Added as optional.** `cuhk_manga_inpaint`; requires MangaInpainting repo + checkpoints; line map auto-generated. |
| **Qwen-Image-Edit** | Semantic/appearance editing, text removal | **Added** `qwen_image_edit`; Diffusers QwenImageInpaintPipeline (Qwen/Qwen-Image-Edit); heavy. |
| **FcF / LDM / SD Inpaint** | Diffusion options | Slower; can give better detail. |

### Recommendations for This Project

- **Best quality (GPU):** **lama_large_512px** (already in project) or **lama_mpe**.
- **Fast / default:** **AOT** or **lama_mpe**.
- **CPU / lightweight:** **PatchMatch** or **opencv-tela**.
- **Optional LaMa ONNX (general):** **lama_onnx** for opencv/inpainting_lama ONNX (512×512); set model_path.
- **Optional RePaint (DDPM):** **repaint** for diffusion inpainting (e.g. google/ddpm-ema-celebahq-256).
- **Optional LaMa (pip):** **simple_lama** inpainter when `simple-lama` or `simple-lama-inpainting` is installed.
- **Optional SD (Diffusers):** **diffusers_sd_inpaint** when `diffusers` and `accelerate` are installed; prompt-based inpainting.
- **Optional FLUX Fill:** **flux_fill** for high-quality inpainting (12B); enable CPU offload if needed.
- **Optional Kandinsky:** **kandinsky_inpaint** for Kandinsky 2.1 inpainting via Diffusers.
- **Optional SDXL (quality):** **diffusers_sdxl_inpaint** for 1024 inpainting when quality matters more than speed; same Diffusers stack as SD.
- **Optional DreamShaper:** **dreamshaper_inpaint** for DreamShaper 8 inpainting (512); quality-focused alternative to SD 1.5 inpainting.
- **Optional SD2 768:** **diffusers_sd2_inpaint** for Stable Diffusion 2 inpainting at 768; middle ground between SD 1.5 (512) and SDXL (1024).
- **Optional Fluently v4:** **fluently_v4_inpaint** for anime/comic inpainting (Diffusers); good for manga bubbles when using diffusion.
- **Optional CUHK Manga:** **cuhk_manga_inpaint** for Seamless Manga Inpainting; set repo path and checkpoints (see [MangaInpainting](https://github.com/msxie92/MangaInpainting)); line map is auto-generated.

### 3.1 Inpainting for manga/manhua/manhwa – Hugging Face & best options (research)

Research focus: **text removal** in manga, manhua, manhwa (speech bubbles, sound effects). Prefer models that **fill from context** (LaMa-style) over **generative** (diffusion) to avoid squiggly/artifact fills.

| Source | Model | Type | Notes |
|--------|--------|------|--------|
| **Hugging Face** | **dreMaz/AnimeMangaInpainting** | LaMa (PyTorch .ckpt) | **Already used** in project as `lama_large_512px`. Finetuned on 300k manga/anime; better than older lama_mpe on manga. Best HF option for manga text removal. |
| **Hugging Face** | **mayocream/lama-manga-onnx** | LaMa manga (ONNX) | ONNX export of dreMaz/AnimeMangaInpainting (FourierUnitJIT). Alternative backend if you want ONNX inference. |
| **Hugging Face** | **opencv/inpainting_lama** | LaMa general (ONNX) | **Added** `lama_onnx`; 512×512 ONNX, lightweight; *general* inpainting. For manga prefer dreMaz or lama_manga_onnx. |
| **CUHK / IOPaint** | **Seamless Manga Inpainting (MangaInpainting)** | Two-phase (semantic + appearance) | SIGGRAPH 2021. **Manga-specific**: disentangles structural lines and screentone; better on high-quality manga than LaMa in some tests. Weights: Google Drive (see [MangaInpainting](https://github.com/msxie92/MangaInpainting)); **not on Hugging Face**. Requires **line map** + mask + image (three inputs). Used in IOPaint as `--model manga`. |
| **IOPaint / Lama-Cleaner** | **manga** (Lama-Cleaner) | Same as CUHK MangaInpainting | Same CUHK model; IOPaint auto-downloads. `--model manga` in Lama-Cleaner/IOPaint. Best for manga when you can run IOPaint; not a single HF model. |
| **Diffusion (HF)** | FLUX.1-Fill, SD inpainting, DreamShaper, etc. | Generative | **Not recommended for text removal**: often produce squiggly or text-like artifacts in bubbles. Use LaMa-based models above. |

**Summary – best for manga/manhua/manhwa text removal:**

1. **In project (recommended):** **lama_large_512px** (from dreMaz/AnimeMangaInpainting on HF). Best balance of quality and integration; no extra inputs (image + mask only).
2. **Optional pip:** **simple_lama** or **simple-lama-inpainting** for an alternative LaMa API; good for text removal.
3. **Manga-specific (academic):** **Seamless Manga Inpainting** (CUHK). Best when you have line maps and can run [MangaInpainting](https://github.com/msxie92/MangaInpainting) or IOPaint with `--model manga`; not on HF, requires line extraction step.
4. **ONNX (HF):** **mayocream/lama-manga-onnx** if you want the same manga LaMa in ONNX form (e.g. for other runtimes).
5. **Avoid for speech-bubble text removal:** DreamShaper and other diffusion inpainters on HF; use LaMa-based instead.

**Manhwa / webtoon:** No dedicated HF inpainting model found for Korean webtoon. **dreMaz/AnimeMangaInpainting** (lama_large_512px) and general LaMa work well on manhwa too (similar clean bubbles and flat colors). For colored webtoon, same models apply; FLUX/SD inpainting still risk artifacts.

---

### 3.2 Recommended inpainting settings (lama_large_512px)

For **manga/comic speech-bubble text removal**, **lama_large_512px** (dreMaz/AnimeMangaInpainting) remains the best option in the project and on Hugging Face: LaMa large finetuned on 300k manga/anime, image+mask only, no line map required.

| Setting | Recommended | Notes |
|--------|-------------|--------|
| **Inpainter** | **lama_large_512px** | Best balance for manga; optional: flux_fill (higher quality, diffusion), AOT, patchmatch (CPU). |
| **inpaint_size** | **512** or **1024** | 512 = less VRAM, gentler on small bubbles; 1024 = more detail on large regions. Avoid 2048 unless needed (risk of artifacts). |
| **mask_dilation** | **2** (default) or **3–5** | 3×3 kernel. 2 = balanced; 3–5 = more coverage for dots/smudges; 0–1 = minimal distortion on tiny bubbles. |
| **precision** | **bf16** (GPU) / **fp32** (CPU) | bf16 when supported for speed and VRAM. |

**Alternatives (when to use):**

- **FLUX.1-Fill** (`flux_fill`): Benchmarks show top inpainting quality; 12B params, gated HF. Use if you have VRAM and want to try; can leave faint artifacts in bubbles (diffusion). Enable CPU offload if needed.
- **Fluently v4** (`fluently_v4_inpaint`): SD-based, anime/comic style; **in project**. Same diffusion caveat as DreamShaper for text removal.
- **CUHK Manga Inpainting** (`cuhk_manga_inpaint`): **In project.** Better on high-quality manga; set MangaInpainting repo path and checkpoints; line map is auto-generated (simple Canny-based). Or use IOPaint `--model manga` externally.
- **ProPainter**: Video inpainting (ICCV 2023); not for single-image manga bubbles.

**Bottom line:** For manga text removal there is no clearly better single model on Hugging Face than **lama_large_512px**. Tune **inpaint_size** and **mask_dilation** (0–5) for your pages; use **flux_fill** only if you want to experiment with diffusion and accept possible artifacts.

---

## 4. Summary: What’s in the Project vs Added

| Category | Already in project | Added / documented |
|----------|--------------------|----------------------|
| **OCR** | Surya, Manga OCR, Paddle OCR, Paddle VL (server), PaddleOCRVLManga, OneOCR, mit32/48px, Windows/Mac/Google/Bing/LLM | **TrOCR**, **GOT-OCR2**, **GLM-OCR**, **Donut**, **PaddleOCR-VL (HF)**, **Qwen2.5-VL/OlmOCR 7B** (`qwen2vl_7b`), **DeepSeek-OCR** (`deepseek_ocr`), **LightOnOCR-2-1B** (`lighton_ocr`), **Chandra 9B** (`chandra_ocr`), **DocOwl2 9B** (`docowl2_ocr`), **Nanonets-OCR2-3B** (`nanonets_ocr`), **Ocean-OCR 3B** (`ocean_ocr`), **InternVL2 8B/2B** (`internvl2_ocr`), **Manga OCR Mobile** (`manga_ocr_mobile`, TFLite), **Nemotron Parse** (`nemotron_ocr`, full-page, bbox assignment). |
| **Text detection** | CTD, Paddle det, EasyOCR det, YSG, Stariver | **MMOCR** (`mmocr_det`), **PP-OCRv5** (`paddle_det_v5`), **Surya det** (`surya_det`), **Magi** (`magi_det`), **TextMamba** (`textmamba_det`, stub until official code), **CRAFT** (`craft_det`), **HF object-detection** (`hf_object_det`, default model_id = ogkalu/comic-text-and-bubble-detector); see recommendations above. |
| **Inpainting** | opencv-tela, PatchMatch, AOT, lama_mpe, lama_large_512px | **Simple LaMa**, **Diffusers SD**, **SD2 768**, **SDXL 1024**, **DreamShaper**, **Fluently v4**, **CUHK Manga**, **FLUX Fill**, **Kandinsky**, **RePaint** (`repaint`), **LaMa ONNX** (`lama_onnx`), **Qwen-Image-Edit** (`qwen_image_edit`), **MAT** (`mat`, repo+checkpoint); see recommendations above. |

---

## 5. Additional Models Researched (Nemotron, Manga OCR Mobile, NuMarkdown, Text-Classification)

### 5.1 NVIDIA Nemotron Parse v1.1

| Aspect | Details |
|--------|---------|
| **Link** | [nvidia/NVIDIA-Nemotron-Parse-v1.1](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1) |
| **Type** | Document parsing / layout + OCR (image → structured text + bboxes + classes) |
| **Architecture** | ViT-H vision encoder + mBART decoder, &lt;1B params; bfloat16. |
| **Input** | Full-page image (min 1024×1280, max 1648×2048) + prompt. |
| **Output** | Single string with markdown/LaTeX + bboxes + semantic classes (title, section, caption, table, etc.); postprocessing extracts `classes`, `bboxes`, `texts`. |
| **Use case** | PDF/PPT extraction, document understanding, tables/formulas. |
| **Fit for BallonsTranslator** | **Low.** Pipeline is full-page document → one structured output. We need per-**region** OCR (crop → text per `TextBlock`). Nemotron does detection+recognition+layout in one shot; our app already does detection (CTD) then OCR per box. Could be used as an alternative “full page OCR” mode (one call per page, then map output bboxes to our blocks), but that would be a different workflow and duplicate detection. **Not recommended** for current per-block OCR slot. |

### 5.2 Manga OCR Mobile (bluolightning)

| Aspect | Details |
|--------|---------|
| **Link** | [bluolightning/manga-ocr-mobile](https://huggingface.co/bluolightning/manga-ocr-mobile) |
| **Type** | Lightweight Japanese manga OCR (mobile/edge). |
| **Training** | PyTorch → converted to **TFLite** (AI Edge Torch); ~10M params. |
| **Performance** | ~7.4% CER, ~73% exact-match on Manga109s; comparable to PaddleOCR-VL-For-Manga. Struggles with English letters and punctuation. |
| **Inference** | TFLite (mobile/edge); technical docs and GitHub repo referenced on HF. |
| **Fit for BallonsTranslator** | **Medium (optional).** Same “manga speech bubble” use case as existing **Manga OCR** (kha-white). Differences: (1) **TFLite** not PyTorch — would need `tflite_runtime` and a separate code path; (2) lighter/faster, good for low-resource devices; (3) Japanese-focused, worse on English. **Recommendation:** Only add if you explicitly want a lightweight/TFLite option or mobile deployment; otherwise existing **manga_ocr** (PyTorch) is simpler and already integrated. |

### 5.3 NuMarkdown-8B-Thinking (numind)

| Aspect | Details |
|--------|---------|
| **Link** | [numind/NuMarkdown-8B-Thinking](https://huggingface.co/numind/NuMarkdown-8B-Thinking) |
| **Type** | Reasoning OCR VLM: document image → thinking + answer (full-page Markdown). |
| **Architecture** | Qwen 2.5-VL-7B fine-tune, ~8B params; Transformers + optional vLLM. |
| **Output** | Full document as Markdown (tables, structure); reasoning tokens can be 20%–500% of answer length. |
| **Use case** | Document→Markdown for RAG, complex layouts/tables; benchmarks beat GPT-4o, OCRFlux. |
| **Fit for BallonsTranslator** | **Low.** Same pipeline mismatch as Nemotron: full-page → one Markdown document. We need **per-region** text for each `TextBlock` (crop → one string per block). NuMarkdown is also heavy (8B, reasoning), so slower and more VRAM than Surya/Manga OCR. **Not recommended** for the current “OCR module” slot; could be a separate “export page as Markdown” feature if desired. |

### 5.4 Text-Classification Models (Hugging Face trending)

| Aspect | Details |
|--------|---------|
| **Link** | [Models – pipeline: text-classification, sort: trending](https://huggingface.co/models?pipeline_tag=text-classification&sort=trending) |
| **Type** | Text classification (e.g. sentiment, topic, intent), not OCR or detection. |
| **Fit for BallonsTranslator** | **N/A for OCR/detection.** Could be used in a **post-OCR** step to classify text (e.g. dialogue vs caption vs sound effect) for different translation or styling — not replacing or implementing OCR/detection. No change to current OCR/detection implementation recommended from this pipeline. |

### 5.5 Summary: Implement?

| Model | Implement as OCR? | Note |
|-------|-------------------|------|
| **Nemotron Parse v1.1** | No | Full-page document parser; doesn’t match per-block OCR API; different workflow. |
| **Manga OCR Mobile** | **Yes** (`manga_ocr_mobile`) | TFLite; optional deps. Lightweight Japanese manga OCR. |
| **NuMarkdown-8B-Thinking** | No | Full-page doc→Markdown; heavy; wrong abstraction for per-block OCR. |
| **Text-classification (trending)** | N/A | Use only for optional post-OCR classification, not for OCR/detection. |

---

## 6. GOT-OCR2 and Hugging Face Text Detection

### 6.1 GOT-OCR2 (Implemented)

| Aspect | Details |
|--------|---------|
| **Link** | [stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf), [Transformers doc](https://huggingface.co/docs/transformers/model_doc/got_ocr2) |
| **Type** | Unified end-to-end OCR (plain/scene/formatted text, tables, formulas, region-level). |
| **Size** | 580M params; vision encoder + long-context decoder. |
| **Input** | Image (or batch); for per-block use we pass each crop as one image. Also supports `box=[x1,y1,x2,y2]` or color for region-level on full image. |
| **Output** | Plain text (default) or formatted (markdown/tikz/smiles) via prompt. |
| **In project** | **Added (full)** as OCR module `got_ocr2`: single + batched inference, `stop_strings`, bf16/fp16, configurable batch size. Use for high-quality unified OCR (documents, comics). |

### 6.2 Hugging Face Text Detection

| Aspect | Details |
|--------|---------|
| **Finding** | HF has no standard **scene-text-only detection** (image → bboxes) in the same way as CTD or Paddle det. Layout/document models (LayoutLM, DocLayout-YOLO, etc.) typically expect or output layout + text together, not raw bboxes from image only. |
| **LightOnOCR-2-1B-bbox** | VL model that outputs text *and* bounding boxes; different pipeline (end-to-end spotter), not a drop-in replacement for “detector then OCR”. |
| **Recommendation** | Keep using **CTD** and **Paddle det** for text detection. Add HF-based detection only if a clear “image → boxes” model appears and fits the `(mask, List[TextBlock])` API. |

### 6.3 GLM-OCR (Implemented)

| Aspect | Details |
|--------|---------|
| **Link** | [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) |
| **Type** | Lightweight 0.9B document OCR (text, formula, table); chat-style API. |
| **In project** | **Added** as OCR module `glm_ocr`. Use for document/mixed content; requires transformers with GLM-OCR support (e.g. 5.x or model-specific). |

### 6.4 Donut (Implemented)

| Aspect | Details |
|--------|---------|
| **Link** | [naver-clova-ix/donut-base-finetuned-docvqa](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa), [donut-base-finetuned-cord-v2](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2) |
| **Type** | OCR-free document understanding; image + task prompt → text. DocVQA = Q&A / read text; CORD = receipt/form parsing. |
| **In project** | **Added** as OCR module `donut`. Use DocVQA with prompt “What is the text in this image?” for generic read, or CORD for structured extraction. |

### 6.5 Stable Diffusion Inpainting via Diffusers (Implemented)

| Aspect | Details |
|--------|---------|
| **Link** | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting), [Diffusers Inpainting](https://huggingface.co/docs/diffusers/using-diffusers/inpaint) |
| **Type** | Prompt-based diffusion inpainting; white mask = fill region. |
| **In project** | **Added** as inpainter `diffusers_sd_inpaint`. Optional; install `diffusers` and `accelerate`. Slower than LaMa; good for natural-looking fill. |

### 6.6 PaddleOCR-VL via Hugging Face Transformers (Implemented)

| Aspect | Details |
|--------|---------|
| **Link** | [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL), [Transformers doc](https://huggingface.co/docs/transformers/model_doc/paddleocr_vl) |
| **Type** | SOTA 0.9B document parsing; 109 languages; text, table, formula, chart. Element-level recognition via chat prompt “OCR:”. |
| **In project** | **Added** as OCR module `paddleocr_vl_hf`. Requires transformers 5.x for PaddleOCR-VL architecture. Use for best multilingual document OCR from HF. |

### 6.7 FLUX.1-Fill Inpainting (Implemented)

| Aspect | Details |
|--------|---------|
| **Link** | [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), [Diffusers Flux](https://huggingface.co/docs/diffusers/api/pipelines/flux) |
| **Type** | 12B rectified flow; inpainting and outpainting; no strength parameter. |
| **In project** | **Added** as inpainter `flux_fill`. Optional; install `diffusers` and `accelerate`. Use CPU offload if VRAM is limited. |

### 6.8 Kandinsky 2.1 Inpainting (Implemented)

| Aspect | Details |
|--------|---------|
| **Link** | [kandinsky-community/kandinsky-2-1-inpaint](https://huggingface.co/kandinsky-community/kandinsky-2-1-inpaint) |
| **Type** | CLIP + diffusion prior; prompt-based inpainting; white mask = fill. |
| **In project** | **Added** as inpainter `kandinsky_inpaint`. Optional; same deps as other Diffusers inpainters. |

### 6.9 Stable Diffusion 2 Inpainting 768 (Implemented)

| Aspect | Details |
|--------|---------|
| **Link** | [stabilityai/stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) |
| **Type** | SD2 inpainting at 768×768; quality option between SD 1.5 (512) and SDXL (1024). |
| **In project** | **Added** as inpainter `diffusers_sd2_inpaint`. Optional; same Diffusers stack; default inpaint_size 768. |

### 6.10 Nanonets-OCR2-3B (Implemented)

| Aspect | Details |
|--------|---------|
| **Link** | [Nanonets/Nanonets-OCR2-3B](https://huggingface.co/Nanonets/Nanonets-OCR2-3B) (or equivalent on Hugging Face) |
| **Type** | 3B VLM OCR; document/scene text via chat-style API; quality-focused. |
| **In project** | **Added** as OCR module `nanonets_ocr`; HF transformers; messages + image (path or PIL); apply_chat_template then generate. |

### 6.11 Ocean-OCR 3B (Implemented)

| Aspect | Details |
|--------|---------|
| **Link** | [guoxy25/Ocean-OCR](https://huggingface.co/guoxy25/Ocean-OCR), [arXiv:2501.15558](https://arxiv.org/abs/2501.15558) |
| **Type** | 3B MLLM; first to outperform dedicated OCR (PaddleOCR, TextIn); document, scene, handwritten. |
| **In project** | **Added** as OCR module `ocean_ocr`; HF AutoModelForCausalLM; trust_remote_code; bind_processor + model.processor + generate. |

### 6.12 InternVL2 8B/2B (Implemented)

| Aspect | Details |
|--------|---------|
| **Link** | [OpenGVLab/InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B), [InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B) |
| **Type** | VLM series 2B–8B; OCRBench 794, DocVQA 91.6; document/chart/scene text. |
| **In project** | **Added** as OCR module `internvl2_ocr`; HF AutoModel + chat(); dynamic_preprocess + transform; trust_remote_code. |

### 6.13 Surya text detection (Implemented)

| Aspect | Details |
|--------|---------|
| **Link** | [vikp/surya_det](https://huggingface.co/vikp/surya_det), [surya-ocr PyPI](https://pypi.org/project/surya-ocr/) |
| **Type** | Line-level text detection; 90+ languages; Segformer; document/comic. |
| **In project** | **Added** as detector `surya_det`; uses surya-ocr `batch_text_detection` + load_model/load_processor. |

### 6.14 Manga Whisperer – Magi (Implemented)

| Aspect | Details |
|--------|---------|
| **Link** | [ragavsachdeva/magi](https://huggingface.co/ragavsachdeva/magi), [GitHub](https://github.com/ragavsachdeva/magi), CVPR 2024 |
| **Type** | Unified manga: panels, text boxes, characters, reading order, dialogue association. |
| **In project** | **Added** as detector `magi_det`; text boxes only (use any OCR after); HF AutoModel, trust_remote_code; requires einops. |

### 6.15 TextMamba (Placeholder)

| Aspect | Details |
|--------|---------|
| **Link** | [arXiv:2512.06657](https://arxiv.org/abs/2512.06657) – TextMamba: Scene Text Detector with Mamba |
| **Type** | Scene text detection; Mamba SSM; CTW1500 89.7%, TotalText 89.2%. |
| **In project** | **Placeholder** `textmamba_det`; raises clear error until official code is released. |

---

## 7. References

- PaddleOCR-VL: [arXiv:2510.14528](https://arxiv.org/abs/2510.14528), [Hugging Face](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- Ocean-OCR: [arXiv:2501.15558](https://arxiv.org/abs/2501.15558)
- Surya: [GitHub](https://github.com/VikParuchuri/surya), [PyPI surya-ocr](https://pypi.org/project/surya-ocr/)
- MAT: [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Li_MAT_Mask-Aware_Transformer_for_Large_Hole_Image_Inpainting_CVPR_2022_paper.html), [GitHub](https://github.com/fenglinglwb/MAT)
- LaMa: Resolution-robust large mask inpainting with Fourier convolutions
- TextMamba: [arXiv](https://arxiv.org/html/2512.06657v1)
- CoMix (comic benchmark): [arXiv:2407.03550](https://arxiv.org/abs/2407.03550)
- CodeSOTA OCR benchmarks: [codesota.com/ocr](https://codesota.com/ocr)
- GOT-OCR2: [Transformers](https://huggingface.co/docs/transformers/model_doc/got_ocr2), [stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
- GLM-OCR: [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), [Transformers GLM-OCR](https://huggingface.co/docs/transformers/model_doc/glm_ocr)
- Diffusers Inpainting: [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)
- Donut: [naver-clova-ix/donut-base-finetuned-docvqa](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa), [Donut paper](https://arxiv.org/abs/2111.15664)
- FLUX.1-Fill: [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
- Kandinsky 2.1 Inpaint: [kandinsky-community/kandinsky-2-1-inpaint](https://huggingface.co/kandinsky-community/kandinsky-2-1-inpaint)
