# Hard-subtitle external tools → models, OCR, detection → BalloonsTranslator

This document is based on the **actual code and READMEs** of the projects you listed. It answers: **what detection/OCR stack each tool uses**, **what can be integrated into BT**, and **what BT modules already cover the same ideas**.

---

## Summary matrix

| Project | Text **detection** (how subs are found) | **OCR / recognition** | Shippable as a BT Python module? |
|--------|----------------------------------------|------------------------|----------------------------------|
| [SubtitleOCR](https://github.com/nhjydywd/SubtitleOCR) | **Closed-source** engine (DLL + C API); UI only open ([custom.md](https://github.com/nhjydywd/SubtitleOCR/blob/main/custom.md)). README credits **PaddleOCR**. | Same binary stack; “自研模型” only in **paid** tier. | **No** — cannot import or redistribute their algorithm binary under BT’s module system. |
| [MioSub](https://github.com/corvo007/MioSub) | Not a classical “text detector”: **ASR** (e.g. Whisper-class) + alignment + LLM. | **Speech-to-text**, not burned-in glyph OCR. | **No** as a detector module; different problem (**audio** vs **pixels**). Use BT video ASR path if you need speech subs. |
| [Hardcoded-Subtitles-Extractor](https://github.com/kaushalag29/Hardcoded-Subtitles-Extractor) | **FFmpeg** extracts frames; **bottom crop** (scripted). No ML detector in repo. | Native **`./OCR` binary** in [do-ocr.py](https://github.com/kaushalag29/Hardcoded-Subtitles-Extractor/blob/main/do-ocr.py) — macOS-oriented ([glowinthedark/macOCR](https://github.com/glowinthedark/macOCR) / Apple-style OCR). Optional **GPT** cleanup in app flow. | **No** unique model; the OCR binary is **not** a cross-platform Python library. |
| [Phr33d0m/subtitle-tools](https://github.com/Phr33d0m/subtitle-tools) **`ocrp`** | Delegates to **[VideOCR](https://github.com/devmaxxing/videocr-PaddleOCR)** ([ocrp.md](https://github.com/Phr33d0m/subtitle-tools/blob/main/docs/ocrp.md)): **fixed crop** + **brightness threshold** + frame dedup (`similar-image`, `similar-pixel`). Under the hood that project uses **PaddleOCR** for text localization + recognition. | **PaddleOCR** (via VideOCR CLI). | **Concepts** yes (crop, brightness gate, dedup); **engine** already available in BT as **`paddle_det` / `paddle_v5` + OCR`** or **`rapidocr`**, without shelling out to `videocr.py`. |
| [Subtitles-remover](https://github.com/Emam546/Subtitles-remover) | **User-drawn ROI** + **HSV/color range** mask (OpenCV). | **Inpainting** to remove region; not OCR-focused. | **No** new model. Same idea as BT **skip detect + bottom band** + **color/threshold masks** + **LaMa** / other inpainters. |

---

## Per-repo technical notes (from source)

### SubtitleOCR (望言OCR)

- **Models:** Proprietary **cxx** libs + packaged **models** (see custom.md: copy `cxx-libs` and `models`).
- **Public credit:** [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).
- **BT alignment:** Use **`paddle_v5`** / **`paddle_det`** + **`paddle_ocr`** (or **`rapidocr`**) for an open, pip-installable stack in the same family.

### MioSub

- **Stack:** Electron/TypeScript app; **Gemini**, **Whisper**/whisper.cpp, **FFmpeg**, forced alignment, burn-in (see repo README).
- **BT alignment:** Video translator **ASR / translation** features; not a replacement for **hardcoded** text detection.

### Hardcoded-Subtitles-Extractor

- **do-ocr.py:** Spawns `./OCR` with locale + flags; **parallel** `ThreadPoolExecutor` over PNG snaps — no Python OCR API.
- **Detection:** Effectively **fixed spatial crop** + time sampling from shell pipeline (`do-all.sh`), not a learned detector.
- **BT alignment:** **Region preset / ROI** + **`videocr_subtitle_det`** or **neural det** + **`rapidocr`** / **`paddle`** / **`videocr_tesseract`**.

### subtitle-tools → `ocrp` + VideOCR

From [ocrp.py](https://github.com/Phr33d0m/subtitle-tools/blob/main/ocrp.py) / [ocrp.md](https://github.com/Phr33d0m/subtitle-tools/blob/main/docs/ocrp.md):

- **Backend:** Path to **`videocr.py`** (e.g. [devmaxxing/videocr-PaddleOCR](https://github.com/devmaxxing/videocr-PaddleOCR)).
- **Typical CLI knobs:** `--crop`, `-b` / **brightness threshold** (default **145**), `-c` conf, `-s` similarity, `--similar-image`, `--similar-pixel`, `--skip`, language `ch`, GPU flag.
- **Meaning:** **PaddleOCR**-based text boxes + recognition, with **classical CV gating** (brightness) and **temporal dedup** of similar frames.
- **BT already has:** OCR **cache** keyed by geometry + content hash (`video_translator`); neural **det**; configurable **bottom region**. What we do **not** mirror 1:1 is a single global **“brightness only”** gate identical to VideOCR — closest is tuning **`videocr_subtitle_det`** thresholds or using **Paddle/RapidOCR** on a crop.

### Subtitles-remover

- **Mask:** Color range in ROI + inpaint (OpenCV + Python sidecar per README).
- **BT alignment:** **Inpainter** modules + mask from detector or fixed band; no OCR extraction focus.

---

## What is **implementable** inside BalloonsTranslator (no closed binaries)

| Idea from external tools | Status in BT |
|--------------------------|--------------|
| **PaddleOCR** det + rec | **`paddle_det`**, **`paddle_v5`**, Paddle-backed OCR modules |
| **RapidOCR** (ONNX PP-OCR style) | **`rapidocr`** det + **`rapidocr`** OCR |
| **Tesseract** on cropped bands | **`videocr_tesseract`** |
| **OpenCV** threshold / morphology / Canny subtitle band | **`videocr_subtitle_det`** (incl. `edge_link_subtitle`, `hybrid_stroke_edge`, **`detection_output: full_band`**) |
| **Fixed crop** / bottom fraction | Video translator **region preset**, **`skip_detect`**, **`detect_roi`** paths |
| **Skip redundant frames** for OCR | Per-block **OCR cache** + optional **scene / band** skipping in video pipeline |
| **LLM cleanup** of OCR lines | **Flow fixer** / translator-adjacent correctors where configured |
| **Native macOS OCR binary** | **Not** integrated — platform-specific and not a library |
| **SubtitleOCR / VideOCR CLI as subprocess** | Possible as a **personal script**, not recommended as a bundled default (paths, GPL coupling, maintenance) |

---

## Practical recommendation (hardcoded text on screen)

1. **Default away from pure CV** if strokes/fades are hard: **`rapidocr`** or **`paddle_v5`** (det + rec).
2. **Stay on `videocr_subtitle_det`:** `input_color_order=bgr`, `threshold_mode=edge_link_subtitle` or `hybrid_stroke_edge`, **`detection_output=full_band`**, tune `full_band_col_trim` / Canny params as needed.
3. **Optional external tool:** Run **[VideOCR](https://github.com/devmaxxing/videocr-PaddleOCR)** or **SubtitleOCR** **outside** BT for extraction-only, then import SRT — orthogonal to BT’s internal modules.

---

## License / distribution

Embedding **SubtitleOCR’s** engine, **AGPL** stacks like **MioSub**, or redistributing **macOCR** binaries would impose different obligations than the current **optional pip** model (Paddle, RapidOCR, etc.). This doc assumes users install those deps themselves, consistent with existing BT modules.

---

## BallonsTranslator: CTD + cnOCR and video inpaint

In the **Video translator** pipeline, **CTD** (or any detector) supplies **masks/blocks**; **cnOCR** only runs on those crops. **Inpainting** is independent of that pairing: it still receives OpenCV **BGR** frames, so the same **BGR→RGB** conversion and **subtitle mask expansion** (descenders / halos) apply whether you use CTD+cnOCR or a subtitle-tuned detector.

**cnOCR / cnstd log spam:** If `det_model_name` was **db_mobilenet_v3_small**, cnstd often tries **ONNX first**, fails, then falls back to PyTorch (warnings). Prefer **ch_PP-OCRv5_det** in cnOCR settings; saved configs with the old value are migrated automatically.

**LaMa ONNX + wide subtitle crops:** The OpenCV LaMa ONNX path used to **stretch every crop to 512×512**, which badly warps horizontal text (often looks fine on the top of strokes and wrong on the bottom). It now **letterboxes** (uniform scale + pad) before inference. Per-block inpainting also **expands polygon masks vertically** (`inpaint_block_mask_vertical_expand`) because CTD quads often miss descenders and halos that the full-frame video dilate never reached.

**Video translator + `lama_mpe`:** With **Inpaint → full image** **off** (default), the pipeline passes **detected text blocks** into the inpainter. `InpainterBase` then runs LaMa on each block’s **enlarged crop** only (`inpaint_enlarge_ratio`), not on the whole frame or the whole bottom band. Turn **full image** on only if you explicitly want one full-frame inpaint pass using the dilated mask.

**`inpaint_enlarge_ratio` near the bottom of the frame:** `enlarge_window` in `utils/imgproc_utils.py` **clips the crop to the image**—it cannot extend past the bottom edge. So for bottom subtitles, extra ratio mostly adds context **above** the line (there is little or no room below). A **very large** ratio can (1) **slow** each block (bigger crops), (2) pull in **more scene** above the sub, which sometimes causes **visible structure or color bleed** into the bar if the shot is busy, and (3) make **feathered paste seams** span a wider blend zone. For typical bottom bands, **~2.0** is a good default; raise it mainly when punctuation/halos are still missed at the **sides** or **top** of the box, not “as high as possible” by default.

**Subtitle bar (no inpaint):** Enable **“Subtitle bar: solid box behind text (no inpaint)”** in the Video translator UI (`video_translator_subtitle_black_box_mode`). Detection/OCR/translation run as usual; **inpainting is skipped**. Burn-in draws a **filled rectangle** per detected block using the same **wrap/layout** as the white subtitle text (`_compute_subtitle_block_draw_bbox` / `_draw_text_on_image`), so **multi-line translations grow the bar** behind all lines—not a full-width bottom 20% blackout. Padding and BGR color: `video_translator_subtitle_black_box_padding`, `video_translator_subtitle_black_box_b/g/r`. Timed burn-in (two-pass / SRT timeline) uses the same bar when the option is on (`_draw_timed_subs_on_image` + `_draw_single_style_subtitle`).

---

## References

- [SubtitleOCR](https://github.com/nhjydywd/SubtitleOCR) · [custom.md (binary engine)](https://github.com/nhjydywd/SubtitleOCR/blob/main/custom.md)  
- [MioSub](https://github.com/corvo007/MioSub)  
- [Hardcoded-Subtitles-Extractor](https://github.com/kaushalag29/Hardcoded-Subtitles-Extractor) · [do-ocr.py](https://github.com/kaushalag29/Hardcoded-Subtitles-Extractor/blob/main/do-ocr.py)  
- [Phr33d0m/subtitle-tools](https://github.com/Phr33d0m/subtitle-tools) · [ocrp.py](https://github.com/Phr33d0m/subtitle-tools/blob/main/ocrp.py) · [ocrp.md](https://github.com/Phr33d0m/subtitle-tools/blob/main/docs/ocrp.md)  
- [VideOCR (PaddleOCR)](https://github.com/devmaxxing/videocr-PaddleOCR) (referenced by ocrp)  
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)  
- [Subtitles-remover](https://github.com/Emam546/Subtitles-remover)  
