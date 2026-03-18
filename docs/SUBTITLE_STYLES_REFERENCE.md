# Subtitle sizes, styles & outline – reference

Summary of common standards and practices (Netflix, broadcast, fansubs) for sizing, fonts, and outlines.

## Font

| Source | Recommendation |
|--------|----------------|
| **Netflix** | Arial (or proportional sans-serif). White. |
| **General** | Sans-serif for legibility; avoid serif, thin, small caps. |
| **Anime fansubs** | Clean sans-serif, no decorative elements; real italics; bold often helps. |

**This project:** Arial Bold / Tahoma / Verdana on Windows; DejaVu/Liberation Bold on Linux (see `_subtitle_font_paths()` in `modules/video_translator.py`). White fill, black outline.

---

## Size

| Source | Recommendation |
|--------|----------------|
| **Netflix** | Size **relative to resolution**; target **42 characters per line** (Latin). No fixed pixel values; TTML uses percentage (e.g. 100%). |
| **Anime (Aegisub)** | Script resolution 1280×720 or 1920×1080; size so lines don’t become 3-line subs and remain readable. |
| **General** | “Big friendly letters”; for TV (2–3 m viewing) often **larger than you think**. |

Rough 1080p reference: font size in the **~35–55 px** range (depending on font) often gives ~40–45 characters per line when the line is ~85% of width. Sizing by **% of frame height** (e.g. 3–5% of height) keeps proportions across resolutions.

**This project:** Scale factor × frame dimension, clamped (e.g. 18–96 px). Default scale ~0.038 → ~41 px at 1080p. Styles: default, anime (slightly larger), documentary (slightly smaller).

---

## Outline / border / stroke

| Source | Recommendation |
|--------|----------------|
| **Readability guides** | **~2 px black stroke** around text for legibility on any background. Thicker strokes can look heavy; avoid ultra-thin. |
| **Anime (Aegisub)** | **Border size 1.5–2.5** (font-dependent). Outline color black or dark. Shadow optional with **high transparency** (e.g. alpha 120–200). |
| **Netflix / general** | **Well-defined black border**; optional semi-transparent black shadow for contrast. |

**This project:** 2 px black outline (draw at offsets ±2 px, then fill). No shadow by default; outline only.

---

## Position & layout

| Source | Recommendation |
|--------|----------------|
| **Netflix** | Center-aligned; **bottom or top** of frame; avoid overlap with on-screen text. |
| **Anime (720p)** | Margins ~80–110 px left/right, ~30 px top/bottom. |
| **General** | Max **2 lines** per subtitle; break at natural phrase boundaries. |

**This project:** Bottom placement with safe margin (`BOTTOM_SAFE_FRAC`); center-aligned; wrap by width and by **max 42 characters per line** (`MAX_SUBTITLE_CHARS_PER_LINE`) when wrapping.

---

## Summary table

| Aspect | Common practice | This project |
|--------|-----------------|--------------|
| Font | Arial / sans-serif, white | Arial Bold (or similar), white |
| Size | Relative; ~42 chars/line; readable at distance | Scale × resolution, 18–96 px clamp |
| Outline | 1.5–2.5 px black border | 2 px black outline |
| Shadow | Optional, subtle, transparent | Not used |
| Position | Bottom/top, centered | Bottom, centered, safe margin |
| Lines | Max 2 | Wrapped by width + 42 chars/line (often 1–2 lines) |
| Italic/bold | Optional (e.g. off-screen) | Translator/flow fixer may use *italic* and **bold** in text; rendered on-screen. |

---

## References

- Netflix: [Timed Text Style Guide – General Requirements](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617), [Subtitle Templates](https://partnerhelp.netflixstudios.com/hc/en-us/articles/219375728) (font: white, Arial; size: relative, 42 chars/line).
- Readability: e.g. “2-pixel black stroke” and outline vs. shadow (documentary/accessibility guides).
- Anime typesetting: [Creating Styles (unanimated)](https://unanimated.github.io/ts/ts-styles.htm) (border 1.5–2.5, black outline, shadow alpha 120–200).
- Aegisub: script resolution = video resolution (e.g. 1920×1080) for correct scaling.
