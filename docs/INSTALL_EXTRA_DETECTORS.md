# Optional text detectors and detection+OCR summary

## Which text detection models work best with **none_ocr**?

**none_ocr** means “do not run OCR on the detected regions”; the app keeps whatever text the **detector** already wrote into each block. So **none_ocr** is only useful when the detector is a **spotter** (detection + recognition) that fills `blk.text`.

| Detector | Fills text? | Use with none_ocr? |
|----------|-------------|--------------------|
| **stariver_ocr** | ✅ Yes (API returns boxes + text) | ✅ **Best** – full pipeline in one API call. |
| **hunyuan_ocr_det** | ✅ Yes (full-image spotting) | ✅ **Best** – use none_ocr to keep spotter text. |
| **swintextspotter_v2** | ✅ Yes (when demo outputs text) | ✅ Good – set repo_path; use none_ocr if demo provides text. |
| paddle_det_v5, paddle_det, surya_det, easyocr_det, ctd, dptext_detr, mmocr_det, etc. | ❌ No (detection only) | ❌ Text stays empty – pair with an OCR instead. |

**Recommendation:** For “detection only, no OCR step”, use a spotter: **stariver_ocr**, **hunyuan_ocr_det**, or **swintextspotter_v2**, and set OCR to **none_ocr**. For detection-only models, always choose an OCR (e.g. paddle_rec_v5, surya_ocr, easyocr_ocr, mmocr_ocr).

---

## Detection vs OCR coverage (summary)

| Module / family | Detection | OCR | Notes |
|-----------------|-----------|-----|--------|
| **HunyuanOCR** | ✅ **hunyuan_ocr_det** (full-image spotting → boxes + text) | ✅ **hunyuan_ocr** (per-crop) | Use hunyuan_ocr_det + **none_ocr** to keep spotter text; or + hunyuan_ocr to re-OCR crops. |
| **PaddleOCR v5** | ✅ **paddle_det_v5** | ✅ **paddle_rec_v5** | Pair both for full PP-OCRv5 pipeline. |
| **PaddleOCR (full)** | ✅ **paddle_det** | ✅ **paddle_ocr** | Full pipeline; can also use paddle_det + other OCRs. |
| **Surya** | ✅ **surya_det** | ✅ **surya_ocr** | Detection-only + recognition on crops. |
| **EasyOCR** | ✅ **easyocr_det** | ✅ **easyocr_ocr** | Detection-only + recognition on crops; pair for full EasyOCR. |
| **Stariver (API)** | ✅ **stariver_ocr** (detector; API returns boxes + text) | ✅ **stariver_ocr** (OCR; same name, per-crop API) | Detector fills text → use **none_ocr** to keep it. |
| **SwinTextSpotter v2** | ✅ **swintextspotter_v2** (optional repo) | ✅ When demo outputs text we fill `blk.text` | Set **repo_path**. Use **none_ocr** if demo provides text. |
| **DPText-DETR** | ✅ **dptext_detr** (optional repo) | ❌ Detection only | Pair with any OCR. |
| **MMOCR** | ✅ **mmocr_det** (DBNet etc.) | ✅ **mmocr_ocr** (SAR/CRNN etc.) | Pair both for full MMOCR pipeline. Same deps: mim install mmengine mmcv mmdet mmocr. |
| **Ocean-OCR** | ❌ No (MLLM recognition only) | ✅ **ocean_ocr** | Use any detector (e.g. surya_det, paddle_det_v5) + ocean_ocr. |
| **CTD, Magi, TextMamba, YSG, CRAFT, etc.** | ✅ Various detectors | ❌ Detection only | Pair with any OCR (e.g. surya_ocr, paddle_rec_v5, mmocr_ocr). **craft_det** = standalone CRAFT (pip: craft-text-detector). |

---

## Optional detectors: SwinTextSpotter v2 and DPText-DETR

These detectors are **optional** and require cloning external repositories. Set the **repo_path** parameter to use them.

---

## SwinTextSpotter v2

End-to-end scene text spotting (detection + recognition). IJCV 2025.

- **Repo:** [mxin262/SwinTextSpotterv2](https://github.com/mxin262/SwinTextSpotterv2)
- **In BallonsTranslator:** Text detector **swintextspotter_v2**. Set **repo_path** to your clone (e.g. `C:\repos\SwinTextSpotterv2`).

### Setup

1. Clone the repo and install dependencies (see repo README):
   ```bash
   git clone https://github.com/mxin262/SwinTextSpotterv2.git
   cd SwinTextSpotterv2
   pip install -r requirements.txt  # or follow their install instructions
   ```
2. Download weights from their Model Zoo / README.
3. In BallonsTranslator, select detector **swintextspotter_v2** and set **repo_path** to the clone path.

**Note:** The detector runs the repo’s demo script (e.g. `demo/demo.py`) via subprocess. If the script’s CLI or output format differs, you may need to adjust `modules/textdetector/detector_swintextspotter_v2.py` to match (e.g. argument names, output JSON path).

---

## DPText-DETR

Transformer-based scene text detection. AAAI 2023. **Detection only** (no recognition); use with any OCR.

- **Repo:** [ymy-k/DPText-DETR](https://github.com/ymy-k/DPText-DETR)
- **In BallonsTranslator:** Text detector **dptext_detr**. Set **repo_path** to your clone.

### Setup

1. Clone and install (see repo README):
   ```bash
   git clone https://github.com/ymy-k/DPText-DETR.git
   cd DPText-DETR
   pip install -r requirements.txt
   ```
2. Download pretrained weights as described in the repo.
3. In BallonsTranslator, select detector **dptext_detr** and set **repo_path** to the clone path.

**Note:** The detector looks for `demo.py` or `tools/demo.py` or `eval.py` and runs it on a temp image. If the repo’s inference interface or output format is different, adapt `modules/textdetector/detector_dptext_detr.py` (CLI args, output file path, JSON structure).

---

## Summary

| Detector           | Repo / install              | Type             | Repo path param |
|--------------------|-----------------------------|------------------|----------------|
| swintextspotter_v2 | mxin262/SwinTextSpotterv2    | Spotter (det+rec)| **repo_path**   |
| dptext_detr        | ymy-k/DPText-DETR            | Detection only   | **repo_path**   |
| craft_det          | pip: craft-text-detector     | Detection only   | —               |
| hf_object_det      | Hugging Face model_id        | Detection only   | **model_id**    |

Both are best-effort integrations; exact CLI and output format depend on the upstream repos and may require small changes in the detector modules.

---

## CRAFT (craft_det)

Standalone **CRAFT** text detection (curved/scene text) without EasyOCR. Use with any OCR.

- **In BallonsTranslator:** Text detector **craft_det**.
- **Install:** `pip install craft-text-detector torch`
- **No repo path:** Model is loaded from the package. Select **craft_det** in the detector list and pair with e.g. **surya_ocr**, **paddle_rec_v5**, or **manga_ocr**.

**Dependency conflict:** `craft-text-detector` requires **opencv-python&lt;4.5.4.62**; the main app uses **opencv≥4.8**. If you see a pip conflict or craft_det fails, use **easyocr_det** or **mmocr_det** instead, or see [docs/OPTIONAL_DEPENDENCIES.md](OPTIONAL_DEPENDENCIES.md).

---

## HF object-detection (hf_object_det)

Generic detector using any **Hugging Face object-detection** model (DETR, RT-DETR, etc.). Use when a comic/text detector is published on HF.

- **In BallonsTranslator:** Text detector **hf_object_det**.
- **Install:** `pip install transformers torch`
- **Parameters:** Set **model_id** to the HF repo (e.g. `facebook/detr-resnet-50`). Use **score_threshold** and optionally **labels_include** (comma-separated) to filter. For ogkalu or other comic detectors on HF, set their **model_id** when available.
