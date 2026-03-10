# Troubleshooting

This document covers common issues: **GPU OOM**, **HuggingFace gated models**, **provider API keys**, and **dependency conflicts**. See also [Optional dependency conflicts](OPTIONAL_DEPENDENCIES.md) for module-specific workarounds.

---

## 1. GPU out-of-memory (OOM)

**Symptoms:** CUDA/ROCm out-of-memory errors, crash during detection/OCR/inpainting, or "CUDA error: out of memory".

| What to try | Notes |
|-------------|--------|
| **Reduce batch / size** | Use smaller **detect_size** (e.g. 1024), **inpaint_size**, or disable parallel translation. Config → DL Module → set device or module params. |
| **Tiled inpainting** | Config → Inpainting → **Inpaint tile size** (e.g. 512 or 1024) and **overlap** (e.g. 64). Reduces peak VRAM for large images. |
| **Load model on demand** | Config → DL Module → **Load model on demand**. Models load only when a pipeline runs; frees VRAM when idle. |
| **Unload after idle** | Config → DL Module → **Unload models after idle** (e.g. 5–10 min). Frees VRAM when you leave the app idle. |
| **PYTORCH_ALLOC_CONF** | Already set by `launch.py`: `max_split_size_mb:512` to reduce fragmentation. You can tune or set before launch: `set PYTORCH_ALLOC_CONF=max_split_size_mb:512` (Windows) or `export PYTORCH_ALLOC_CONF=max_split_size_mb:512` (Linux/macOS). |
| **Close other GPU apps** | Browsers, other ML tools, or games can hold VRAM. Close them and retry. |
| **CPU for some stages** | Set **device** to CPU for detector, OCR, or inpainter in Config → DL Module to move that stage off GPU. On 11 GB GPUs with GPU-heavy OCR (e.g. qwen35), running the **detector (hf_object_det)** on CPU avoids OOM; the detector will also auto-fallback to CPU for YOLO if GPU runs out of memory during load or predict. |
| **Inpaint full image** | If per-block inpainting OOMs, try Config → Inpainting → **Inpaint full image** (uses one big pass; can still be heavy but sometimes more stable). |

---

## 2. HuggingFace gated models

**Symptoms:** "401 Unauthorized", "Repository not found", or prompt to log in when downloading models (e.g. LLaMA, some OCR/detector models).

| What to do | Notes |
|------------|--------|
| **Accept terms** | On [huggingface.co](https://huggingface.co), open the model page (e.g. `meta-llama/Llama-2-7b`) and accept the license/terms if required. |
| **HF token** | Create a token: Hugging Face → Settings → Access Tokens → New. Use **Read** (or higher if you need write). |
| **Where to set** | **Config → General → HuggingFace token**, or set env var `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) before running. Prefer env var so the token is not stored in config. |
| **Faster downloads (Xet)** | If `HF_TOKEN` is set, `launch.py` can enable `HF_XET_HIGH_PERFORMANCE=1` for faster first-time downloads. See `utils/model_manager.py`. |
| **Gated model in optional module** | Some detectors/OCRs (e.g. HF object detection, certain VLMs) pull gated models. Same steps: accept terms, set token, then run again. |

---

## 3. Provider API keys and translator/OCR

**Symptoms:** "Invalid API key", "401", "403 Forbidden", or "quota exceeded" when using a translator or cloud OCR.

| Provider / area | What to check |
|-----------------|----------------|
| **Where to set** | Config → DL Module → **Translator** (or OCR) → open the module (e.g. LLM_API_Translator, ChatGPT, Google) → fill **API key** / **Key** in params. Keys are stored in `config.json`; keep the file private. |
| **Format** | Paste the key as given (no extra spaces). OpenAI keys often start with `sk-`; Google/others vary. If the UI has a "Test" button (e.g. **Test translator**), use it to verify. |
| **OpenAI** | Key from [platform.openai.com](https://platform.openai.com/api-keys). Ensure the account has credits and the model you chose is available. |
| **Google / Gemini** | Use an API key from Google AI Studio or Cloud. Check quota and that the model name in the dropdown matches your access. |
| **OpenRouter** | Key from [openrouter.ai](https://openrouter.ai). Free models (ID ending in `:free`) have strict rate limits; see **§4 OpenRouter free-tier 429** below. |
| **DeepL / others** | Key from the provider’s dashboard; set in the same Translator/OCR params. |
| **Proxy** | If behind a proxy: Config → Translator (or module) → **Proxy** (e.g. `http://127.0.0.1:7897`). See README "Translation context" for proxy format. |
| **Rate limits / quota** | "Too many requests" or "quota exceeded" → wait, or switch model/provider, or check the provider’s usage page. For OpenRouter free models, see §4. |

---

## 4. OpenRouter free-tier 429 (rate limit)

**Symptoms:** `Error code: 429`, "temporarily rate-limited upstream", or "Provider returned error" when using an OpenRouter model whose ID ends in `:free` (e.g. `meta-llama/llama-3.3-70b-instruct:free`).

**Documented limits (as of 2024–2025):**

| Limit | Value | Source |
|-------|--------|--------|
| **Requests per minute (RPM)** | 20 | [OpenRouter docs](https://openrouter.ai/docs/api/reference/limits) |
| **Requests per day (RPD)** | 200 (no credits); higher if you have purchased credits | Same |
| **429 from "upstream"** | Even under 20 RPM, the **upstream provider** (e.g. Venice, Z.AI) can be globally rate-limited; OpenRouter returns 429 in that case too | [OpenRouter examples #11](https://github.com/OpenRouterTeam/openrouter-examples/issues/11) |

**Recommended LLM_API_Translator settings for free models:**

| Param | Recommended | Reason |
|-------|-------------|--------|
| **Delay** | 3.5–5 s | 60 ÷ 20 RPM = 3 s minimum between requests; 3.5–5 s leaves headroom and reduces upstream 429s. |
| **Max requests per minute** | 6–10 | Stays under OpenRouter’s 20 RPM; lower = safer. |
| **Rate limit delay** | 60–90 s | When you get 429, wait this long before retry so upstream can recover. |

**Where to set:** Config → DL Module → **Translator** → **LLM_API_Translator** → **Delay**, **Max requests per minute**, **Rate limit delay**.

**Other options:** Use a paid model (no `:free`), add your own OpenRouter key and purchase credits for higher limits, or use **model fallbacks** (multiple models) so the app can switch when one is rate-limited.

---

## 5. Dependency conflicts

**Symptoms:** `pip` reports conflicting versions when installing, or a specific detector/OCR/inpainter fails to import or run.

| What to do | Notes |
|------------|--------|
| **Optional modules** | See **[docs/OPTIONAL_DEPENDENCIES.md](OPTIONAL_DEPENDENCIES.md)** for known conflicts (e.g. **craft_det** with opencv, **simple_lama** with Pillow). Use the suggested alternatives or a separate venv. |
| **Don’t install everything** | Install only the dependencies for the modules you use. Extra pip packages (e.g. `craft-text-detector`, `simple-lama-inpainting`) can conflict with main `requirements.txt`. |
| **Fresh venv** | `python -m venv venv`, activate, then `pip install -r requirements.txt` and `python launch.py`. Reduces conflicts from other projects. |
| **Torch version** | `launch.py` installs PyTorch (CUDA or ROCm) automatically. To force a version, set **TORCH_COMMAND** (e.g. `pip install torch==... torchvision==... --index-url ...`) before running. See README or "Portable setup" for platform notes. |

---

## 6. First run seems stuck or very slow

**Symptoms:** After "Choose models to download" you see "Checking connectivity to the model hosters..." and the app appears to hang for one or more minutes; or downloads are slow.

| What to do | Notes |
|------------|--------|
| **Normal on first run** | The first launch downloads model files (hundreds of MB to over 1 GB depending on packages). You should see progress lines like "downloading data/models/...". Let it finish. |
| **Skip connectivity check** | If the connectivity check takes too long (e.g. firewalled or slow DNS), set **DISABLE_MODEL_SOURCE_CHECK=True** before running: `set DISABLE_MODEL_SOURCE_CHECK=True` (Windows CMD) or `export DISABLE_MODEL_SOURCE_CHECK=True` (Linux/macOS), then `python launch.py`. Some download backends use this to skip pre-download reachability checks. |
| **Text style warning** | If you see "Text style does not exist" on first run, it is harmless: the app creates `config/textstyles/default.json` and continues. |

---

## 7. Pipeline caches, CBR, batch report, manual mode

**OCR and translation caches (Config → DL Module):**

| Option | What it does |
|--------|--------------|
| **Enable OCR cache** | Reuses OCR results for the same image/model/language in the current session. Reduces redundant OCR runs when re-running or changing only translation. |
| **Translation cache** | Reuses translation results for the same source text and settings (when deterministic). Saves API cost on re-runs. |
| **Clear OCR and translation caches** | Tools → Models → **Clear OCR and translation caches**. Clears in-session caches so the next run recomputes. |
| **Release model caches** | Tools → Models → **Release model caches**. Unloads detector/OCR/inpainter/translator models and frees GPU/RAM. |
| **Release model caches after batch** | Config → General. When on, models are unloaded automatically after each full pipeline run. |
| **Manual mode** | Config → General. When on, **Run** processes only the current page (comic-translate style). |

**Opening CBR (RAR comic archives):** Use **File → Open CBR ...** for `.cbr`/`.rar` files. Requires `pip install rarfile` and **WinRAR** or **7-Zip** (with UnRAR) in your system PATH. If it fails, the app shows a message with these requirements.

**Batch report:** If pages were skipped during a run (e.g. soft translation failure), a **Batch report** may open automatically. Use **Tools → Project → Show last batch report** to open it again; double-click a row to jump to that page.

**Run OCR or translation on selected pages:** In the page list (left), right-click selected pages → **Run OCR on selected pages**, **Run translation on selected pages**, or **Run inpainting on selected pages**. Runs only that stage on the selected pages; uses caches.

---

## 8. Tips: comic-style bubbles and detector

For **comic-style speech bubbles** (bubble + text regions), you can use the **Hugging Face object-detection** detector with a model that outputs both bubbles and text:

- **Config → DL Module → Detector** → choose **hf_object_det** (or similar).
- Set **Model ID** to e.g. `ogkalu/comic-text-and-bubble-detector` (or another model that predicts both bubble and text regions).
- In the detector params, set **Labels to include** so that both `bubble` and `text_bubble` (or the model’s label names) are included. This lets the pipeline treat bubbles as first-class regions for layout and inpainting.

See [COMIC_TRANSLATE_RESEARCH.md](COMIC_TRANSLATE_RESEARCH.md) for more detector and layout notes.

## Quick reference

| Issue | First step |
|-------|------------|
| OpenRouter 429 / free tier | Config → Translator → LLM_API_Translator: **Delay** 3.5–5 s, **Max requests per minute** 6–10, **Rate limit delay** 60–90 s. See §4. |
| Translation overflows bubble | Config → General → Typesetting: **Text in box** = Auto fit to box, **Auto layout** on. See §9. |
| GPU OOM | Load model on demand, unload after idle, or lower detect_size / inpaint_size / use tiled inpainting. |
| HF 401 / gated | Accept model terms on huggingface.co, create HF token, set in Config → General or `HF_TOKEN`. |
| Translator/OCR "invalid key" | Set API key in Config → DL Module → that module’s params; use Test button; check proxy if needed. |
| Pip conflict / import error | See [OPTIONAL_DEPENDENCIES.md](OPTIONAL_DEPENDENCIES.md); use a clean venv and only install deps for modules you use. |
| First run "stuck" / slow | Downloads take several minutes; set `DISABLE_MODEL_SOURCE_CHECK=True` to skip long connectivity check if needed. See §6 above. |
| CBR open fails | Install `pip install rarfile` and add WinRAR or 7-Zip (UnRAR) to PATH. See §7. |
| Batch report / skipped pages | Tools → Project → Show last batch report; double-click row to open page. See §7. |

---

## 9. Translation text overflows bubble or formats badly

**Symptoms:** After translation, the text box resizes and extends outside the speech bubble, or text is poorly formatted (wrong line breaks, too big/small). Text may also be cropped at the bottom or form a narrow vertical column.

**What helps:** Use **Auto layout** and **Text in box** = **Auto fit to box** so layout uses the balloon region for line breaks and font scaling. The layout system prefers fewer, longer lines and fuller width usage; when **Constrain text box to bubble** is on, it scales font down if needed so text fits without cropping.

**Settings that help:**

| Where | Setting | Recommendation |
|-------|---------|----------------|
| **Config → General → Typesetting** | **Text in box** | Set to **Auto fit to box** so the program scales font size to fit the balloon. |
| **Config → General → Typesetting** | **Auto layout** | Leave **on** so translation is split into lines according to the balloon region. |
| **Config → General → Typesetting** | **Font Size** | **Decide by program** lets layout choose font size; use **use global setting** only if you want a fixed size. |
| **Config → General → Typesetting** | **Constrain text box to bubble** | Keep **on** so the box stays inside the bubble; layout will scale font down if content would overflow. |
| **config.json** (optional) | `module.layout_optimal_breaks` | Keep `true` (default) for better line breaks (fewer, longer lines). |
| **config.json** (optional) | `module.layout_collision_check` | Keep `true` (default) so layout retries when text would overflow. |

**Per-block:** Select one or more text blocks → right-click → **Format → Auto fit font size to box** to scale font so text fits the current box.

**If it still overflows:** The bubble region comes from the detector mask. Try a different **Text detection** module or increase **box_padding** slightly so the detected region fully contains the bubble; then re-run Detect and Translate.

---

## 10. Text boxes in wrong position, stacked at top-left, or outside the image

**Symptoms:** After translation or layout, text boxes are all at the top-left, or some boxes appear far outside the image.

| What to check | Notes |
|---------------|--------|
| **Constrain text box to bubble** | Config → General → Typesetting → **Constrain text box to bubble**. When **on**, the box is forced to the detected bubble region (with correct image coordinates). If issues persist, try turning it **off** to see if layout without constrain works. |
| **Initial upscale** | Config → General → **Initial upscale** (image_upscale_initial). When **on**, detection runs on a 2× (or larger) image; block coordinates are scaled back at the end of the pipeline. If the run was interrupted (e.g. before inpainting finished), blocks can stay in upscaled coordinates and appear in the wrong place. **Try turning Initial upscale off** to test whether positions fix. |
| **center_text_in_bubble** | If present in `config.json` under `module`, it is **ignored** (the feature was removed). Safe to delete or leave as-is. |
| **Merge gap ratio** | In `config.json`, `module.merge_nearby_blocks_gap_ratio` should be a normal value (e.g. `1.5`). A value like `0.999...` can be a float artifact; set to `1.5` if you use merge nearby blocks. |

**Code safeguard:** When **Constrain text box to bubble** is on, the layout now clamps the text box to the image bounds so it never extends outside the panel, even if the bubble region is wrong or there is an upscale/coordinate mismatch.
