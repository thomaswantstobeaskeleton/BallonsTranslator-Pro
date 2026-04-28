# Models reference: detection, OCR, translation, inpainting

This document maps the **curated list of 20+ models per category** (from community recommendations) to **what BallonsTranslator already supports** and what is optional or potential. Use it to choose modules and plan integrations.

**Target:** Windows + NVIDIA GPU. VRAM notes are approximate. For **quality/accuracy rankings** (best to worst) of all detection, OCR, and translation modules, see **[docs/QUALITY_RANKINGS.md](QUALITY_RANKINGS.md)**.
For tier labels (Stable/Beta/Experimental/External dependency heavy) and cross-platform compatibility, use the canonical matrix: **[docs/MODULE_COMPATIBILITY_MATRIX.md](MODULE_COMPATIBILITY_MATRIX.md)**.

---

### ChatGPT comic detection recommendations (quick reference)

| Recommendation | Status | How to use / note |
|----------------|--------|--------------------|
| **1) ogkalu/comic-text-and-bubble-detector** (Transformer/RT-DETR) | ✅ **Default for hf_object_det** | Would require a new detector that loads this Hugging Face model. **ysgyolo** can load Ultralytics RT-DETR **.pt** checkpoints if the model path contains `rtdetr` (e.g. `ysgyolo_rtdetr_something.pt` in `data/models`), but the ogkalu transformer-based “comic-text-and-bubble-detector” is not a drop-in .pt. |
| **2) ogkalu/comic-speech-bubble-detector-yolov8m** | ✅ **Supported via ysgyolo** | Download the YOLOv8 medium weights from [Hugging Face](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m), save as `data/models/ysgyolo_comic_speech_bubble_v8m.pt` (or any name starting with `ysgyolo`), then select that path in **ysgyolo** detector. This is **not** an YSG (淫書館) model; **YSG** refers only to [YSGforMTL/YSGYoloDetector](https://huggingface.co/YSGforMTL/YSGYoloDetector). |
| **3) mayocream/comic-text-detector-onnx** | ⚠️ **Optional** | Use **ctd** with device=CPU and **custom_onnx_path** to your ONNX file. Default CTD ONNX when empty. |

---

## 1. Text detection & spotting

| Recommended model | In BallonsTranslator | Notes |
|-------------------|----------------------|--------|
| **SwinTextSpotter v2** | ✅ **swintextspotter_v2** | Optional repo; set **repo_path**. Spotter (det+rec). |
| **DPText-DETR** | ✅ **dptext_detr** | Optional repo; set **repo_path**. Detection only. |
| **Manga/comic YOLO (e.g. ogkalu comic-speech-bubble)** | ✅ **ysgyolo** | Use model `ysgyolo_comic_speech_bubble_v8m.pt` in `data/models` (from ogkalu; not YSG series). **YSG (淫書館)** = [YSGforMTL/YSGYoloDetector](https://huggingface.co/YSGforMTL/YSGYoloDetector) only. |
| **CTD (ComicTextDetector)** | ✅ **ctd** | Built-in. detect_size up to 2400. Optional **custom_onnx_path** for alternate ONNX (e.g. mayocream). |
| **PaddleOCR det** | ✅ **paddle_det**, **paddle_det_v5** | Full pipeline with paddle_rec / paddle_rec_v5. |
| **Surya detection** | ✅ **surya_det** | Pair with surya_ocr. |
| **EasyOCR detection** | ✅ **easyocr_det** | Pair with easyocr_ocr. |
| **MMOCR (DBNet etc.)** | ✅ **mmocr_det** | Pair with mmocr_ocr. Same deps: mim install mmengine mmcv mmdet mmocr. |
| **HunyuanOCR spotting** | ✅ **hunyuan_ocr_det** | Full-image spotting; use with none_ocr or hunyuan_ocr. |
| **Stariver (API)** | ✅ **stariver_ocr** | Detector returns boxes+text; use with none_ocr. |
| **Magi, TextMamba, YSG (淫书馆)** | ✅ **magi_det**, **textmamba_det**, **ysgyolo** | **textmamba_det**: stub (official code not yet released; raises clear error). **magi_det**, **ysgyolo**: detection only; pair with any OCR. **YSG (淫书馆)** = [YSGforMTL/YSGYoloDetector](https://huggingface.co/YSGforMTL/YSGYoloDetector) by lhj5426 (19 months from data to training). Same author’s earlier YOLOv8: [ogkalu/manga-text-detector-yolov8s](https://huggingface.co/ogkalu/manga-text-detector-yolov8s). The **ysgyolo** detector can also load other comic YOLO .pt (ogkalu, Kiuyha, etc.); those are not "YSG series". |
| **DBNet / PAN / PSENet** | ✅ (via **mmocr_det**) | MMOCR includes DBNet and other backbones. |
| **CRAFT, TextSnake, SAM-backboned, DocLAYNET, etc.** | ✅ **craft_det**, **hf_object_det**, **mmocr_det** (TextSnake) | **hf_object_det**: default model_id = ogkalu/comic-text-and-bubble-detector. **craft_det**: CRAFT. **mmocr_det**: DBNet/TextSnake. SAM/DocLAYNET not integrated. |
| **TextHawk2, TextMonkey** | ❌ Not integrated | LVLM/OCR-free; would need new spotter/OCR modules. |
| **RT-DETR-Manga, MANGA109 detectors** | ⚠️ Partial | YOLO-style comics detectors work with **ysgyolo** if weights are in `data/models` (e.g. ysgyolo_*.pt). |

---

## 2. OCR / text recognition

| Recommended model | In BallonsTranslator | Notes |
|-------------------|----------------------|--------|
| **HunyuanOCR-1B** | ✅ **hunyuan_ocr** | SOTA lightweight; use with any detector. |
| **PaddleOCR-VL / PP-OCRv5** | ✅ **paddle_rec_v5**, **paddle_ocr**, **PaddleOCRVLManga**, **paddleocr_vl_hf**, **paddle_vl** | Multiple variants; pair with paddle_det / paddle_det_v5. |
| **InternVL 2** | ✅ **internvl2_ocr** | Strong OCR; 8B/2B/4B. |
| **Qwen2-VL / Qwen2.5-VL** | ✅ **qwen2vl_7b** | Vision-language OCR. |
| **DeepSeek OCR** | ✅ **deepseek_ocr** | API/local. |
| **TrOCR** | ✅ **trocr** | Transformer recognition. |
| **Surya OCR** | ✅ **surya_ocr** | Pair with surya_det. |
| **EasyOCR** | ✅ **easyocr_ocr** | Pair with easyocr_det. |
| **MMOCR (SAR/CRNN)** | ✅ **mmocr_ocr** | Pair with mmocr_det. |
| **GLM-OCR** | ✅ **glm_ocr** | |
| **GOT-OCR2** | ✅ **got_ocr2** | |
| **LightOn OCR** | ✅ **lighton_ocr** | |
| **Chandra** | ✅ **chandra_ocr** | |
| **DocOwl2** | ✅ **docowl2_ocr** | |
| **Donut** | ✅ **donut** | |
| **Manga-OCR** | ✅ **manga_ocr** | Japanese/manga. |
| **Manga OCR Mobile** (TFLite) | ✅ **manga_ocr_mobile** | Lightweight Japanese manga OCR; encoder+decoder TFLite from bluolightning/manga-ocr-mobile. Optional: tflite-runtime, huggingface_hub, transformers. |
| **Nemotron Parse** (full-page) | ✅ **nemotron_ocr** | Full-page document OCR with bboxes; assigns text to blocks by overlap. nvidia/NVIDIA-Nemotron-Parse-v1.1. |
| **One-OCR** | ✅ **one_ocr** | |
| **Ocean-OCR** | ✅ **ocean_ocr** | MLLM; use with any detector. |
| **Florence-2** | ✅ **florence2_ocr** | Microsoft vision model; base/large. Crop-based OCR. |
| **MiniCPM-o, InternVL 3, OCRFlux, etc.** | ✅ **minicpm_ocr**, **internvl3_ocr**, **ocrflux** | **minicpm_ocr**: MiniCPM-o-2_6/int4; **internvl3_ocr**: InternVL3 1B/2B/8B; **ocrflux**: OCRFlux-3B document OCR. |
| **APIs (Google Vision, Bing, Nanonets)** | ✅ **google_vision**, **bing_ocr**, **nanonets_ocr** | API keys required. |
| **LLM-based OCR** | ✅ **llm_ocr**, **stariver_ocr** | Generic LLM API or Stariver API. |

**OpenRouter vision models for LLM OCR:** When using **llm_ocr** with provider **OpenRouter** (API key from [openrouter.ai](https://openrouter.ai)), you can pick any vision-capable model. **Free vision models** (image in, text out, $0; [full list](https://openrouter.ai/models?fmt=cards&input_modalities=image&max_price=0&output_modalities=text)): `openrouter/free`, `google/gemma-3-4b-it:free`, `google/gemma-3-12b-it:free`, `google/gemma-3-27b-it:free`, `mistralai/mistral-small-3.1-24b-instruct:free`, `nvidia/nemotron-nano-12b-v2-vl:free`, `qwen/qwen3-vl-30b-a3b-thinking`, `qwen/qwen3-vl-235b-a22b-thinking`. Paid examples: `openai/gpt-4o`, `openai/gpt-4o-mini`, `google/gemini-2.0-flash-001`, `google/gemini-1.5-flash`, `google/gemini-1.5-pro`, `qwen/qwen2.5-vl-72b-instruct`, `qwen/qwen3.5-flash-02-23`, `anthropic/claude-sonnet-4`, `anthropic/claude-3-5-sonnet`. [Image inputs](https://openrouter.ai/docs/features/image-inputs), [API models](https://openrouter.ai/api/v1/models).

---

## 3. Translation

| Recommended model | In BallonsTranslator | Notes |
|-------------------|----------------------|--------|
| **GPT-4o / OpenAI** | ✅ **LLM_API_Translator** (provider OpenAI) | Best contextual translation; API key. |
| **Claude / Gemini** | ✅ **LLM_API_Translator** (OpenRouter) or **ChatGPT** | Use OpenRouter or provider endpoints. **Free models:** provider OpenRouter, then pick a free model from the dropdown (e.g. `openrouter/free`, `meta-llama/llama-3.3-70b-instruct:free`, `stepfun/step-3.5-flash:free`). [Full list](https://openrouter.ai/models?fmt=cards&max_price=0&order=most-popular&output_modalities=text&input_modalities=text). |
| **Google Translate API** | ✅ **google** | |
| **DeepL** | ✅ **DeepL**, **DeepL Free**, **DeepLX API** | |
| **M2M-100** | ✅ **m2m100** | Local CTranslate2; many languages. |
| **Sakura** | ✅ **Sakura** | Japanese↔English. |
| **Sugoi** | ✅ **Sugoi** | |
| **NLLB-200 / OPUS-MT / T5 MT** | ✅ **nllb200**, **opus_mt**, **t5_mt** | **nllb200**: 200 languages (HF); **opus_mt**: Helsinki-NLP per-pair; **t5_mt**: prompt-based T5. |
| **Baidu, Youdao, Caiyun, Papago, Yandex** | ✅ **Baidu**, **Youdao**, **Caiyun**, **Papago**, **Yandex** | API/key. |
| **text-generation-webui** | ✅ **text-generation-webui** | Local LLM. |
| **Copy Source / None** | ✅ **None**, **Copy Source** | No translation. |

---

## 4. Inpainting & rendering

| Recommended model | In BallonsTranslator | Notes |
|-------------------|----------------------|--------|
| **LaMa** | ✅ **lama_large_512px**, **lama_mpe**, **simple_lama**, **lama_onnx**, **lama_manga_onnx** | **lama_onnx**: opencv/inpainting_lama ONNX (general). **lama_manga_onnx**: manga ONNX. |
| **AOT** | ✅ **aot** | |
| **FLUX Fill** | ✅ **flux_fill** | |
| **Stable Diffusion Inpaint** | ✅ **diffusers_sd_inpaint**, **diffusers_sdxl_inpaint**, **diffusers_sd2_inpaint** | |
| **Dreamshaper, Kandinsky** | ✅ **dreamshaper_inpaint**, **kandinsky_inpaint** | |
| **Fluently v4** | ✅ **fluently_v4_inpaint** | |
| **CUHK Manga** | ✅ **cuhk_manga_inpaint** | |
| **OpenCV / PatchMatch** | ✅ **opencv-tela**, **patchmatch** | Lighter options. |
| **Qwen-Image-Edit, RePaint, MAT** | ✅ **qwen_image_edit**, **repaint**, **mat** | **qwen_image_edit**: Diffusers. **repaint**: RePaint DDPM. **mat**: MAT (CVPR 2022) via repo + checkpoint (github.com/fenglinglwb/MAT). SAM3 not integrated. |

---

## 5. Recommendation strategy

- **Detection (priority):** Use **ctd** or **paddle_det_v5** for manga; **surya_det** for general docs; **ysgyolo** with comic bubble model for balloon-only. For SOTA spotting, **swintextspotter_v2** or **hunyuan_ocr_det** + **none_ocr** (when compatible).
- **OCR (priority):** **paddle_rec_v5** or **hunyuan_ocr** for quality; **surya_ocr**, **florence2_ocr**, **internvl2_ocr** for alternatives. Use **none_ocr** only with spotters that fill text.
- **Translation (priority):** **LLM_API_Translator** with GPT-4o/Claude/Gemini for best context; **Sakura** for JP↔EN; **DeepL** or **google** for API; **m2m100** for local multilingual.
- **Inpainting:** **lama_large_512px** for most manga; **flux_fill** or **aot** if you prefer.

---

## 6. Adding new models

- **Detectors:** Implement `TextDetectorBase`, `_detect()` returning `(mask, blk_list)`. Register with `@register_textdetectors('name')`. See `modules/textdetector/detector_*.py` and `docs/INSTALL_EXTRA_DETECTORS.md`.
- **OCR:** Implement `OCRBase`, `_ocr_blk_list()` and optionally `ocr_img()`. Register with `@register_OCR("name")`. See `modules/ocr/ocr_*.py`.
- **Translators:** Implement `BaseTranslator`, `_translate()`. Register with `@register_translator('name')`. See `doc/how_to_add_new_translator.md`.
- **Inpainters:** Implement `InpainterBase`. Register with `@register_inpainter('name')`. See `modules/inpaint/`.

VRAM: small OCR/detection ~2–6 GB; large VLMs (Qwen2-VL 7B, InternVL 8B) ~16–24 GB; translation LLMs depend on size and quantization.

---

## 7. Not integrated (reference)

The following remain **not integrated** by design or feasibility:

| Item | Reason |
|------|--------|
| **ogkalu/comic-text-and-bubble-detector** | ✅ **Integrated** as default model_id for **hf_object_det**. |
| **TextHawk2, TextMonkey** | LVLM/OCR-free spotters; would require new spotter/OCR modules and different API. |
| **MAT** (inpainting) | ✅ **Integrated** as **mat** (set repo_path + checkpoint_path to MAT repo and .pth). |
| **SAM3** (inpainting) | Segmentation model, not a drop-in inpainter; no integration. |
| **SAM-backboned, DocLAYNET** (detection) | No detector module yet. TextSnake via **mmocr_det** (det_model=TextSnake). |
| **Manga OCR Mobile** | ✅ **Integrated** as **manga_ocr_mobile** (TFLite; optional deps). |
| **Nemotron Parse, NuMarkdown** | **Nemotron** ✅ **Integrated** as **nemotron_ocr** (full-page, assigns by bbox overlap). NuMarkdown = full-page doc→Markdown; not integrated. |
