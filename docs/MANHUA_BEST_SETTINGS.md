# Best Models & Settings for Manhua

Recommended **text detection**, **OCR**, and **inpainting** choices for **manhua** (Chinese comics), with concrete settings. Manhua uses Chinese text in speech bubbles, captions, and sound effects—similar to manga but with CJK (Chinese) as the primary language.

---

## 1. Text Detection

**Primary (recommended):** **CTD (ComicTextDetector)**  
Best for comic/manga-style bubbles and panels. Use it first.

| Setting | Recommended value | Notes |
|--------|---------------------|--------|
| **Detect size** | `1280` | Higher = better quality, slower. 1024 is faster. |
| **Merge font size tolerance** | `3.0` | Keeps lines in the same bubble merged. Increase (e.g. 3.5) if you get too many small boxes per bubble. |
| **Box score threshold** | `0.42–0.48` | Lower = more boxes (catches small text, more false positives). For manhua, **0.45** is a good default; try **0.42** if CTD misses text. |
| **Min box area** | `0` or `100–200` | 0 = keep all; 100–200 removes tiny noise. |
| **Device** | `cuda` | Use GPU if available. |

**Fallback (when CTD misses text):**  
- **Surya detection** (`surya_det`): line-level, 90+ languages, good for non-bubble or dense text.  
  - **det_score_thresh**: `0.3` (lower to catch more).  
  - **Device**: `cuda`.  
- **Paddle det** (`paddle_det`): strong for Chinese + English document/comic text. CPU-only when used with Ocean OCR.  
  - **Strict bubble mode**: keep **on** for comics (stricter threshold, min area, shrink, aspect filter).  
  - **det_limit_side_len**: `960` (with Ocean OCR, capped at 960 to avoid timeout on CPU).  
  - **language**: `ch`. **det_db_box_thresh**: `0.72` default; `0.75–0.8` in strict mode.  
  - **min_detection_area**: `200` (strict: 250+). **max_aspect_ratio**: `10`. **box_shrink_px**: `4` (strict: 5).  
  - **merge_same_line_only** and **merge_line_overlap_ratio** `0.35` (strict: 0.5) so different bubbles are not merged.

**Optional (manga-style layout):**  
- **Magi** (`magi_det`): panels, text boxes, reading order. Use if you want layout-aware detection; then use any OCR.

---

## 2. OCR (Text Recognition)

For **manhua**, you want models that handle **Chinese** (Simplified/Traditional) and mixed Chinese + English well.

### Recommended (pick one)

| Model | Best for | Settings |
|-------|----------|----------|
| **Surya OCR** (`surya_ocr`) | 90+ languages, strong on Chinese + mixed | **Language**: `Chinese (Simplified)` or `Chinese + English`. **Fix Latin misread**: `True` (reduces Wg→王 type errors). **Crop padding**: `6–8`. **Batch size**: `16` (reduce to 8 if OOM). **Device**: `cuda`. |
| **Paddle OCR** (`paddle_ocr`) | Strong Chinese, lightweight | **Language**: `Chinese & English`. **OCR version**: `PP-OCRv4`. **use_angle_cls**: `True` if you have rotated text. **rec_batch_num**: `6`. **drop_score**: `0.5`. **Device**: `cuda`. |
| **GLM-OCR** (`glm_ocr`) | Lightweight (0.9B), document/comic | **model_name**: `zai-org/GLM-OCR`. **use_bf16**: `True` if your GPU supports it. **Device**: `cuda`. |
| **PaddleOCR-VL (HF)** (`paddleocr_vl_hf`) | SOTA 109-language, document | Use prompt `OCR:`; requires transformers 5.x. **Device**: `cuda`. |
| **Ocean-OCR** (`ocean_ocr`) | High quality doc/scene/handwritten | Heavier (3B); quality over speed. **Device**: `cuda`. |

### Single recommendation for manhua

- **Surya OCR** with **Language** = `Chinese (Simplified)` (or `Chinese + English` if you have mixed text) and **Fix Latin misread** = `True`.  
- If you prefer a lighter, Chinese-tuned option: **Paddle OCR** with **Language** = `Chinese & English`.

---

## 3. Inpainting (Text Removal)

Inpainting removes the original text so you can overlay translations. For manhua, use a good balance of **quality** and **speed**.

### Recommended (pick one)

| Model | Use case | Settings |
|-------|----------|----------|
| **Simple LaMa** (`simple_lama`) | Best balance: good quality, easy install | No extra params; ensure **Device** = `cuda` if available. |
| **lama_large_512px** | Best quality (project LaMa) | **mask_dilation**: `1` (or `0` if still mutilates). **inpaint_size**: `1024` default. Always uses the model (no solid-color fill), so small bubbles get smooth inpainting instead of a grey/colored box. Use GPU. |
| **lama_mpe** | Fast + good quality | Defaults; use GPU. |
| **AOT** | Default/fast inpainting | Good for quick runs. |
| **FLUX.1-Fill** (`flux_fill`) | Highest quality, heavy (12B) | Enable **CPU offload** if you have &lt;12GB VRAM. Slower. |

**Avoid for text removal:** DreamShaper and other SD/diffusion inpainters often produce squiggly or low-quality fills in speech bubbles. Use LaMa-based models above instead.

### Single recommendation for manhua

- **Simple LaMa** or **lama_large_512px** on **cuda** for best quality/speed tradeoff.  
- Use **FLUX.1-Fill** only if you want maximum quality and have the VRAM/time.

---

## 4. Quick setup summary (manhua)

| Step | Choice | Main settings |
|------|--------|----------------|
| **Detection** | CTD | detect_size `1280`, box score `0.45`, merge tolerance `3.0`, device `cuda`. |
| **OCR** | Surya OCR | Language `Chinese (Simplified)` or `Chinese + English`, Fix Latin misread `True`, device `cuda`. |
| **Inpainting** | Simple LaMa or lama_large_512px | device `cuda`. |

If CTD misses some text, add a second pass with **Surya detection** or **Paddle det** (language `ch`), or switch to **Surya detection** as primary for dense/document-like pages.
