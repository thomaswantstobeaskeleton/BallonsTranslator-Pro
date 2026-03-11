# Optional module dependencies and conflicts

Some optional detectors and inpainters have dependency version constraints that conflict with the main `requirements.txt`. This document explains the conflicts and workarounds.

---

## 1. CRAFT detector (`craft_det`)

**Conflict:** The PyPI package **craft-text-detector** (0.4.3) requires:

- `opencv-python>=3.4.8.29,<4.5.4.62`

The project’s main requirements use **opencv-python>=4.8.1.78** (for compatibility with other components). So with the default install, **craft-text-detector** is incompatible.

**What happens:**  
If you install `craft-text-detector` in the same environment, pip may report a dependency conflict. If you then upgrade opencv to satisfy the main requirements, **craft_det** may fail at runtime (import or inference).

**Workarounds:**

1. **Don’t use `craft_det`**  
   Use **easyocr_det** (includes CRAFT) or **mmocr_det** (e.g. TextSnake) instead.

2. **Separate environment for CRAFT**  
   In a dedicated venv, install an older opencv and craft-text-detector:
   ```bash
   pip install "opencv-python>=4.2,<4.5.4.62" craft-text-detector torch
   ```
   Run only the CRAFT-related workflow in this env (or accept running the full app there with older opencv).

3. **Keep main env as-is**  
   Do not install `craft-text-detector`. The **craft_det** option will appear in the UI but will show an error when selected; use another detector.

---

## 2. Simple LaMa inpainter (`simple_lama`)

**Conflict:** The PyPI package **simple-lama-inpainting** (0.1.2) requires:

- `pillow>=9.5.0,<10.0.0`

The project typically uses **Pillow 10.x** (e.g. 10.4.0). So with the default install, **simple-lama-inpainting** is incompatible.

**What happens:**  
Pip may report a dependency conflict. With Pillow 10.x installed, **simple_lama** may fail at import or runtime.

**Workarounds:**

1. **Don’t use `simple_lama`**  
   Use **lama_large_512px**, **lama_mpe**, **lama_onnx**, or **lama_manga_onnx** instead.

2. **Downgrade Pillow (only if you need simple_lama)**  
   ```bash
   pip install "pillow>=9.5,<10"
   ```
   This may affect other features that expect Pillow 10.x.

3. **Separate environment**  
   Use a dedicated venv with `pillow>=9.5,<10` and `simple-lama-inpainting` for the simple_lama inpainter only.

---

## Summary

| Optional module   | Conflicting package        | Conflict                          | Prefer instead                          |
|-------------------|----------------------------|-----------------------------------|-----------------------------------------|
| **craft_det**     | craft-text-detector        | opencv-python &lt;4.5.4.62 vs ≥4.8 | easyocr_det, mmocr_det, ctd             |
| **simple_lama**   | simple-lama-inpainting     | pillow &lt;10 vs 10.x               | lama_large_512px, lama_onnx, lama_manga_onnx |
| **rapidocr_det / rapidocr** | — | None | Install: `pip install rapidocr-onnxruntime` (optional). |
| **nemotron_ocr_v1** | — | Python 3.12+ only | Install: `pip install nemotron-ocr` (optional). |
| **nemotron_parse** | — | None | Install: `pip install transformers accelerate albumentations timm` (optional). |

The main application and all other detectors/inpainters work with the versions in `requirements.txt`. These notes only apply if you explicitly want **craft_det** or **simple_lama**.

---

## 3. RapidOCR detector and OCR (`rapidocr_det`, `rapidocr`)

**No conflict.** Optional package:

- **rapidocr-onnxruntime**

**What it provides:** Text detector **rapidocr_det** and OCR **rapidocr** (ONNX, no GPU required). Good for EasyScanlate-like pipelines and Korean/Chinese/English.

**Install when needed:**

```bash
pip install rapidocr-onnxruntime
```

**Optional:** Place PP-OCRv5 det/rec models in `data/models/rapidocr/` and set paths in Config → DL Module → Text detection (rapidocr_det) / OCR (rapidocr). For **recommended settings** (manhwa/comics, aligned with [EasyScanlate](https://github.com/Liiesl/EasyScanlate)), see **docs/EASYSCANLATE_INTEGRATION.md** § Configuration → Recommended settings.

---

## 4. Nemotron OCR v1 (`nemotron_ocr_v1`)

**Optional package:** **nemotron-ocr** (PyPI).

**What it provides:** OCR option **nemotron_ocr_v1**: full-page detection + recognition (NVIDIA). Runs on the full image and assigns text to your detected blocks by bbox overlap.

**Requirements:** Python **3.12 or 3.13** (the package does not support 3.10/3.11). The rest of the project supports 3.10+.

**Visibility:** When **Config → General → Dev mode** is off, **nemotron_ocr_v1** appears in the OCR dropdown only if your Python is 3.12+. With Dev mode on, it always appears (selecting it on Python &lt;3.12 will show an error at load). **Tools → Manage models** → Check all models lists it with install hint in Details.

**Install when needed (Python 3.12+):**

```bash
pip install nemotron-ocr
```

Models are downloaded from Hugging Face on first run. Config → OCR → **nemotron_ocr_v1**; optional params: merge_level (word/sentence/paragraph), iou_threshold.

---

## 5. Nemotron Parse 1.1 (`nemotron_parse`)

**Optional dependencies:** **transformers**, **accelerate**, **torch**, **albumentations**, **timm** (project already has transformers/torch). Postprocessing is loaded from the model repo **nvidia/NVIDIA-Nemotron-Parse-v1.1** (postprocessing.py) on first use.

**What it provides:** OCR option **nemotron_parse**: full-page document OCR with bboxes and semantic classes. Best for documents; minimum resolution 1024×1280.

**Install when needed:**

```bash
pip install transformers accelerate albumentations timm
```

Config → OCR → **nemotron_parse**; optional params: model_id, device, min_resolution, iou_threshold. Legacy config key **nemotron_ocr** is migrated to **nemotron_parse** on load. It appears in the OCR dropdown when dependencies are available (or in **Dev mode**); **Tools → Manage models** lists it with install hint.

---
