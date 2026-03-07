# Models for conjoined bubble detection

You can **avoid the HF comic detector entirely** by using **YOLO as primary**: set **model_id** to a path to a YOLO `.pt` file (e.g. `data/models/ysgyolo_comic_text_segmenter_v8m.pt`). The detector then uses that YOLO for the main run and never loads `ogkalu/comic-text-and-bubble-detector`. Conjoined YOLOs (and optional HF conjoined) still run as before.

---

The **hf_object_det** detector can run **one or more conjoined models** on the same image. All conjoined outputs are merged; when a primary (merged) box contains at least **conjoined_min_boxes_in_primary** conjoined boxes, that primary box is replaced by those smaller boxes—splitting merged/touching bubbles. **Conjoined boxes that do not fall inside any primary box (e.g. SFX the primary missed) are now added to the final output**, so conjoined models can fill in text outside bubbles.

- **conjoined_backend = yolo** — run only YOLO model(s) from **conjoined_yolo_paths** (one path per line or comma-separated; up to 10).
- **conjoined_backend = hf** — run only Hugging Face object-detection model(s) from **conjoined_model_ids** (comma-separated; up to 5).
- **conjoined_backend = both** — run both YOLO and HF lists and merge results.

Use **conjoined_dedup_iou** (e.g. 0.85) to merge duplicate boxes from different models before the replace logic.

---

## Conjoined YOLO models (work well together)

**Note:** The filename prefix `ysgyolo_` is a convention for "YOLO .pt in data/models". **YSG (淫書館)** refers only to [YSGforMTL/YSGYoloDetector](https://huggingface.co/YSGforMTL/YSGYoloDetector); do not categorize other YOLO models (ogkalu, Kiuyha, etc.) under "YSG series".

Place each `.pt` in `data/models/` with a name starting with `ysgyolo` (or any path). List them in **conjoined_yolo_paths**, one per line or comma-separated. All run; results are merged and deduplicated.

| Model | HF repo | Save as (in data/models/) | Strength |
|-------|---------|---------------------------|----------|
| **ogkalu/comic-text-segmenter-yolov8m** | [link](https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m) | `ysgyolo_comic_text_segmenter_v8m.pt` | Text regions (dialogue, SFX); best for splitting merged text. |
| **ogkalu/comic-speech-bubble-detector-yolov8m** | [link](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m) | `ysgyolo_comic_speech_bubble_v8m.pt` | Speech bubbles; adjacent merged bubbles. |
| **Kiuyha/Manga-Bubble-YOLO** | [link](https://huggingface.co/Kiuyha/Manga-Bubble-YOLO) | `ysgyolo_manga_bubble_nano.pt` (weights/yolo26n.pt), `ysgyolo_manga_bubble_yolo26s.pt` (weights/yolo26s.pt) | Lightweight; single “Text” class = bubbles + text regions; nano fast, small higher recall for text outside bubbles. |
| **huyvux3005/manga109-segmentation-bubble** | [link](https://huggingface.co/huyvux3005/manga109-segmentation-bubble) | `ysgyolo_manga109_seg_bubble.pt` | YOLO11 seg; high-precision bubble boxes. |
| **kitsumed/yolov8m_seg-speech-bubble** | [link](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble) | `ysgyolo_kitsumed_speech_bubble.pt` | Segmentation masks; boxes from .boxes used for conjoined. |
| **deepghs/manga109_yolo** (nano) | [link](https://huggingface.co/deepghs/manga109_yolo) | `ysgyolo_manga109_text_n.pt` (v2023.12.07_n_yv11/model.pt) | Manga109-trained; class **text** (dialogue + SFX-like regions). Use conjoined_labels_include so only text is kept (body/face/frame ignored). |
| **deepghs/AnimeText_yolo** (gated) | [link](https://huggingface.co/deepghs/AnimeText_yolo) | `ysgyolo_animetext_n.pt` (yolo12n_animetext/model.pt) | Anime/manga scene text (class **text_block**). Request access on HF, then add to conjoined. |
| **IncreasingLoss/YoloV11_m_chinese_character_Detector** | [link](https://huggingface.co/IncreasingLoss/YoloV11_m_chinese_character_Detector) | `ysgyolo_chinese_char_m.pt` (yolo_chinese_m.pt) | Chinese character/text detection; useful for **manhua**. Class mapped to text_bubble. |

**conjoined_labels_include:** Use `bubble,text_bubble,text_free` to also keep text outside bubbles (captions, SFX). Use `bubble,text_bubble` for bubbles only. Class names are normalized (e.g. `text` → `text_bubble`, `character`/`char` → `text_bubble`, `onomatopoeia` → `text_free`).

---

## Detecting text outside bubbles (text_free)

Sound effects, captions, and signs outside speech bubbles use the **text_free** label. To detect them well:

1. **Primary:** Keep **model_id** as the comic-text-segmenter (`ysgyolo_comic_text_segmenter_v8m.pt`); it’s trained for general text and is the main source for text_free.
2. **Labels:** Set **labels_include** and **conjoined_labels_include** to `bubble,text_bubble,text_free` (default now includes text_free).
3. **Threshold:** Use a lower **score_threshold_text_free** (e.g. 0.15–0.2) so more outside-bubble text is kept.
4. **Extra conjoined models for text regions:** Add models that output “text” or “text regions” (mapped to text_bubble). The **Kiuyha Manga-Bubble-YOLO** models (nano + small) use a single “Text” class for both bubbles and text regions; adding **both** nano and small improves recall for text outside bubbles.

**Conjoined paths when focusing on text outside bubbles** (paste into conjoined_yolo_paths; set **conjoined_labels_include** = `bubble,text_bubble,text_free`):

```
data/models/ysgyolo_comic_speech_bubble_v8m.pt
data/models/ysgyolo_manga_bubble_nano.pt
data/models/ysgyolo_manga_bubble_yolo26s.pt
data/models/ysgyolo_manga109_seg_bubble.pt
data/models/ysgyolo_kitsumed_speech_bubble.pt
data/models/ysgyolo_manga109_text_n.pt
```

The 6th model (**ysgyolo_manga109_text_n.pt**) is Manga109-trained with a “text” class; with conjoined_labels_include set, only its text boxes are kept (body/face/frame are ignored).

- **Kiuyha small (yolo26s):** Download [Kiuyha/Manga-Bubble-YOLO](https://huggingface.co/Kiuyha/Manga-Bubble-YOLO) → **weights/yolo26s.pt** and save as `data/models/ysgyolo_manga_bubble_yolo26s.pt`. Use together with nano for better coverage of text regions.
- The comic-text-segmenter as primary remains the main source for text_free; conjoined models add redundancy and help split merged bubbles.

**Config tweaks for better SFX / text outside bubbles:**
- **score_threshold_text_free:** set to **0.15** (or 0.1) so more low-confidence SFX/captions are kept.
- **conjoined_score_threshold:** set to **0.25** or **0.3** so conjoined models contribute more borderline text regions.
- **conjoined_min_box_area:** set to **500** or **1000** so small SFX/caption boxes from conjoined are not dropped (default 2500 can filter them).
- **detect_min_side:** set to **1280** so small text isn’t lost on large pages (optional).
- Use **all 6 conjoined paths** below (including **ysgyolo_manga109_text_n.pt**) so the Manga109 “text” model adds another source for text regions.

**Optional (gated):** [deepghs/AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo) is trained for anime/manga scene text (class `text_block`). It is gated: open the link, accept the conditions, then download **yolo12n_animetext/model.pt** and save as `data/models/ysgyolo_animetext_n.pt`. Add that path to conjoined_yolo_paths; `text_block` is mapped to text_bubble.

---

## Best-quality 5-model conjoined setup

For maximum conjoined quality, use **conjoined_backend = yolo** and list **5 YOLO models** in **conjoined_yolo_paths** (one per line). Order does not matter; all run and results are merged.

**Recommended 5 (paste into conjoined_yolo_paths):**

```
data/models/ysgyolo_comic_text_segmenter_v8m.pt
data/models/ysgyolo_comic_speech_bubble_v8m.pt
data/models/ysgyolo_manga_bubble_nano.pt
data/models/ysgyolo_manga109_seg_bubble.pt
data/models/ysgyolo_kitsumed_speech_bubble.pt
```

- **ogkalu text-segmenter** — tight text/SFX boxes.
- **ogkalu speech-bubble** — bubble-only splits.
- **Kiuyha Manga-Bubble** — extra coverage (nano is fast).
- **huyvux manga109-seg** — high-precision bubbles.
- **kitsumed seg** — another bubble/seg source.

Set **conjoined_dedup_iou** to **0.85** so overlapping boxes from different models are merged. **conjoined_min_boxes_in_primary** = 2 (replace only when at least 2 conjoined boxes fall inside a primary box).

If you have only 2–3 of these, use what you have; 2–3 models still improve over a single conjoined model.

---

## Hugging Face models (conjoined_backend = hf or both)

Set **conjoined_model_ids** to comma-separated HF model ids (e.g. `ogkalu/comic-text-and-bubble-detector`). Must be DETR/RT-DETR-style; output `box`, `score`, `label`.

| Model | HF repo | Notes |
|-------|---------|--------|
| **ogkalu/comic-text-and-bubble-detector** | [link](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) | Multi-class (bubble, text_bubble, text_free). Can add as conjoined when also using YOLOs (backend = both). |

---

## Settings

- **enable_conjoined_secondary:** Check to enable.
- **conjoined_backend:** `yolo` | `hf` | `both`.
- **conjoined_yolo_paths:** One path per line or comma-separated (up to 10). E.g. 5 paths for best quality.
- **conjoined_model_ids:** Comma-separated HF ids (up to 5). Optional when backend = hf or both.
- **conjoined_score_threshold:** 0.35–0.4.
- **conjoined_labels_include:** `bubble,text_bubble,text_free` (or empty). Add **text_free** to detect captions/SFX outside bubbles.
- **conjoined_min_boxes_in_primary:** 2.
- **conjoined_min_box_area:** 2500.
- **conjoined_dedup_iou:** 0.85 (merge duplicates when using multiple models); 0 = off.

**Manhwa / Manhua / Chinese:** The **ogkalu** comic-speech-bubble and comic-text-segmenter models are trained on manga, **webtoon, manhua**, and Western comics. For **manhua** (Chinese comics), add **ysgyolo_chinese_char_m.pt** ([IncreasingLoss/YoloV11_m_chinese_character_Detector](https://huggingface.co/IncreasingLoss/YoloV11_m_chinese_character_Detector)) to conjoined_yolo_paths for extra Chinese text coverage.

**Legacy:** Single **conjoined_yolo_path** and **conjoined_model_id** are still read if the multi-value fields are empty.

Install **ultralytics** for YOLO: `pip install ultralytics`.
