# BallonsTranslatorPro — Extended Release

**Manga & comic translation with 20+ detectors, 30+ OCR engines, 15+ inpainters, 370+ fonts, and full docs.**

---

## What’s in this release

This is **BallonsTranslatorPro**, an extended fork of [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) with extra models, safer defaults, and detailed documentation. Everything from the original app still works; new modules appear in the same settings dropdowns once dependencies are installed.

---

## Highlights

- **Text detection:** CTD, Paddle v5, Surya, MMOCR, CRAFT, **HF object detection** (ogkalu comic-text-and-bubble-detector), Magi, DPText-DETR, SwinTextSpotter v2, YSG YOLO, Hunyuan spotter, and more.
- **OCR:** Paddle OCR v5, Surya, HunyuanOCR, TrOCR, Qwen2-VL 7B, InternVL2/3, Florence-2, MiniCPM, OCRFlux, **Manga OCR Mobile** (TFLite), **Nemotron Parse** (full-page), EasyOCR, MMOCR, and 20+ other backends.
- **Inpainting:** LaMa (large 512px, ONNX, manga ONNX), Simple LaMa, AOT, **FLUX Fill**, Diffusers (SD 1.5 / SD2 / SDXL, DreamShaper, Kandinsky), **RePaint**, **Qwen-Image-Edit**, **MAT**, **CUHK Manga**, **Fluently v4**, and more.
- **370+ fonts** included (comic, manga, Blambot, CC fonts, anime-style, and others) for typesetting.
- **Docs:** Quality rankings, model reference, best settings for manhua, optional dependencies, extra detectors setup, and a full modifications guide.

---

## Quick start

1. **Extract** the zip to a folder (e.g. `BallonsTranslator`).
2. **First run:**  
   `python launch.py`  
   This installs base dependencies and downloads default models into `data/`.
3. **Settings:** Open the app’s settings panel and choose **Text detection**, **OCR**, **Inpainting**, and **Translation**. New modules show up automatically; install only the dependencies for the ones you use (see `docs/` and `README_MODIFICATIONS.md`).
4. **Config:** Copy `config/config.example.json` to `config/config.json` if you need a clean config. Add API keys and passwords only in the app (stored in `config.json`, which is not in the zip for safety).

---

## Important notes

- **Python:** 3.10 or 3.11 recommended (avoid Microsoft Store Python).
- **Models:** If the first-run download fails, use the links in the main README (MEGA / Google Drive) and place the `data` folder in the project root.
- **API keys:** Do not put keys in source. Use the app’s settings; keys are saved in `config/config.json`.
- **Optional modules:** Some detectors/OCR/inpainters need extra install steps or have dependency conflicts. See `docs/OPTIONAL_DEPENDENCIES.md` and `docs/INSTALL_EXTRA_DETECTORS.md`.

---

## Documentation (in `docs/` and root)

| File | Description |
|------|-------------|
| **README.md** | Main setup and usage. |
| **README_MODIFICATIONS.md** | Full list of changes, how to run every model, settings (inpaint size, mask dilation, etc.), and fixes. |
| **docs/QUALITY_RANKINGS.md** | Quality tiers and recommendations for detection, OCR, translation. |
| **docs/MODELS_REFERENCE.md** | Map of recommended models to modules in this fork. |
| **docs/MANHUA_BEST_SETTINGS.md** | Suggested settings for Chinese manhua. |
| **docs/OPTIONAL_DEPENDENCIES.md** | Optional module conflicts (e.g. CRAFT, Simple LaMa) and workarounds. |
| **docs/INSTALL_EXTRA_DETECTORS.md** | Optional detectors (SwinTextSpotter, DPText-DETR, CRAFT, HF object det). |

---

## Credits

- Original: [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) by dmMaze. **BallonsTranslatorPro** adds detectors, OCR, inpainters, fonts, settings (Logical DPI, recent limit, confirm before run, dark mode), Tools (region merge, re-run detection only), and docs.

---

**Enjoy translating.**
