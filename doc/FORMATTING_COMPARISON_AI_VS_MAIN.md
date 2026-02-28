# Formatting / layout: BallonsTranslator-ai vs main (ours)

Comparison focused on **webcomics and Chinese manhua**: how each repo does text layout, font scaling, and fitting inside bubbles.

---

## 1. Core layout algorithm (`utils/text_layout.py`)

**Same in both.**  
Both use:

- `layout_lines_aligncenter` / `layout_lines_alignside` for line breaking and positioning
- `layout_text()` with:
  - `ref_src_lines` (reference original line widths)
  - `start_from_top` for center-aligned blocks when top line is much longer than bottom
  - Mask-based checks (border_thr ~220) to keep text inside the bubble

So the **line-breaking and centering logic** that works well for manhua/webcomics is shared.

---

## 2. Where they differ: `layout_textblk()` in `ui/scenetext_manager.py`

This is where “their method” and “our method” diverge. We have **stricter fitting and overflow control**; -ai is more permissive.

### 2.1 Balloon / text area ratio (resize_ratio)

| Aspect | BallonsTranslator-ai | Ours (main) |
|--------|----------------------|-------------|
| Non-CJK (e.g. EN→ZH) | `ballon_area / 1.2 / text_area`, min 0.7 | `ballon_area / 1.70 / text_area`, min 0.4 |
| CJK + ref_src_lines | `resize_ratio_src * 1.5`, floor 0.5 | Same idea, floor 0.4 |
| Resize clamp | `min(max(resize_ratio, 0.6), 1)` | `min(max(resize_ratio, 0.5), 1)` + extra cap for huge blocks |

So we scale down **more aggressively** so text fits inside the bubble with margin; -ai allows slightly larger text (1.2, 0.7) and less aggressive floor (0.6).

### 2.2 Shrinking the text region (region_rect)

| Aspect | BallonsTranslator-ai | Ours (main) |
|--------|----------------------|-------------|
| Region inset | None | Yes: round/oval (ar ≥ 0.5) → 72% of region; elongated → 85% |
| Purpose | — | Keeps text away from bubble edges so it doesn’t touch the curve |

We explicitly shrink the usable region so layout stays inside the bubble; -ai does not.

### 2.3 Overflow: fitting laid-out text into the bubble

| Aspect | BallonsTranslator-ai | Ours (main) |
|--------|----------------------|-------------|
| After layout | No extra fit step | If laid-out size (w,h) > region (rw,rh): `post_resize_ratio = min(rw/w, rh/h) * LAYOUT_FIT_RATIO` (0.80), min 0.50 |
| Effect | Text can overflow bubble | Text is scaled down again so it fits inside with margin |

So we have a **second pass** that guarantees the final text block fits inside the bubble; -ai can overflow.

### 2.4 Font size and block size

| Aspect | BallonsTranslator-ai | Ours (main) |
|--------|----------------------|-------------|
| Min font size | None (can go very small) | `LAYOUT_MIN_FONT_PT = 10.0` |
| Block width | `maxw * 1.5` | `maxw * LAYOUT_WIDTH_STROKE_FACTOR` (1.28) – more room for stroke/outline |
| Block height | `xywh[3]` | `max(xywh[3], line_height) * LAYOUT_HEIGHT_PADDING` (1.06) – avoids last line on the edge |

We avoid tiny unreadable text and add padding for stroke and last line; -ai does not.

### 2.5 Short text in big bubbles (scale up)

| Aspect | BallonsTranslator-ai | Ours (main) |
|--------|----------------------|-------------|
| When region >> text | No scale-up | If `region_area > 2.5 * text_area`: scale up so text fills ~78% of region, cap 1.4× |

We **increase** font size when the bubble is much larger than the text so it doesn’t look tiny; -ai keeps original scale only.

### 2.6 Huge blocks and image bounds

| Aspect | BallonsTranslator-ai | Ours (main) |
|--------|----------------------|-------------|
| Block > 50% of image | No special handling | Bounding rect capped to 50% of image size; resize_ratio capped so text doesn’t dominate page |
| Final position | No clamp | Block clamped so it never goes outside image (x, y kept in [0, im_w-w], [0, im_h-h]) |

We avoid one bubble taking over the panel and keep blocks on-canvas; -ai does not.

### 2.7 Vertical / CJK

- Both disable **auto layout for vertical blocks** (`if blkitem.blk.vertical: return`).
- Both use **ref_src_lines** and **start_from_top** for center-aligned, multi-line source.
- **Vertical rendering** (e.g. `vertical_force_aligncentel` in `scene_textlayout.py`) is the same in both.

So vertical/CJK behavior in layout and rendering is aligned; the main differences are in horizontal fitting and scaling above.

---

## 3. Summary table

| Feature | -ai | Ours | Better for webcomic/manhua |
|---------|-----|------|----------------------------|
| Line layout (center/side, ref_src_lines) | ✓ | ✓ | Same |
| Stricter fit (ballon/text ratio) | Looser (1.2, 0.7) | Stricter (1.70, 0.4) | Ours (less overflow) |
| Region inset (round/elongated) | No | Yes (72% / 85%) | Ours (text inside curve) |
| Overflow post-fit | No | Yes (fit ratio 0.80, min 0.50) | Ours (no clipping) |
| Min font size | No | 10 pt | Ours (readable) |
| Block width/height padding | 1.5×, no height pad | 1.28×, 1.06× height | Ours (stroke + last line) |
| Scale up in big bubbles | No | Yes (up to 1.4×, fill ~78%) | Ours (no tiny text) |
| Huge block cap (e.g. >50% image) | No | Yes | Ours (stable layout) |
| Clamp to image bounds | No | Yes | Ours (no off-canvas) |

---

## 4. Conclusion

- **Core “method” (line breaking, center/side, ref_src_lines, vertical rules)** is the same in both; that’s what gives good structure for webcomics and manhua.
- **Our implementation adds**:
  - Stricter scaling so text stays inside the bubble (and we shrink the region first).
  - An explicit overflow-fit step so the final block never exceeds the bubble.
  - Minimum font size, block padding for stroke/height, scale-up for big bubbles, and caps/clamping for huge blocks and image bounds.

So for **webcomics and Chinese manhua**, our formatting is **stricter and more defensive**: we prioritize “text always inside the bubble and readable” over “allow slightly larger text even if it can overflow.”  

If you want to **nudge our behavior toward -ai** (e.g. slightly larger text, less aggressive fit), we could:

- Increase `ballon_area / 1.70` to something like `1.4` or `1.2`, and/or raise the minimum resize from `0.5` to `0.6`.
- Slightly relax `LAYOUT_FIT_RATIO` (e.g. 0.85) or `LAYOUT_FIT_RATIO_MIN` (e.g. 0.55).

without dropping the overflow-fit, min font, and clamp logic that make our method robust for webcomics and manhua.
