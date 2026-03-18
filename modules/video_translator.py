"""
Video translator: run detect -> OCR -> translate -> inpaint on video frames,
then draw translated text onto the inpainted frame. Used by the Video translator UI.

GitHub / references that inspired options and behavior:
- video-subtitle-remover (YaoFANGUK): config-driven pipeline, sample-every-N, multiple inpainting modes.
- video-subtitle-extractor (YaoFANGUK): ROI / bottom region for subtitles, batch, multi-language OCR.
- EraseSubtitles, Subtitle_Inpainting: detect -> mask -> inpainting; ProPainter/Ultralytics.
- Subtitle_Inpainting_Cognative_Computing: scene detection (PySceneDetect) + CRAFT + temporal inpainting.
- Subtitle quality quantification (Snimm): bottom-of-screen ROI, EasyOCR, bounding box intersection.
Possible future additions: scene-change keyframes only (SSIM/keyframe extraction), temporal smoothing
(flow-guided inpainting), optional FFmpeg for encode/decode.
"""
from __future__ import annotations

import hashlib
import os
import re
import numpy as np
import cv2
from typing import List, Optional, Tuple, Any

from utils.textblock import TextBlock, examine_textblk, remove_contained_boxes, deduplicate_primary_boxes
from utils.config import pcfg
from modules.inpaint.base import build_mask_with_resolved_overlaps
from utils.logger import logger as LOGGER

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Fraction of frame height to keep clear at the bottom so subtitles are not cut off (e.g. by TV overscan or player chrome).
BOTTOM_SAFE_FRAC = 0.08

# Max characters per line (Netflix-style); wrap will try to keep lines at or under this when possible.
MAX_SUBTITLE_CHARS_PER_LINE = 42

# Per-region OCR cache: max entries to avoid unbounded growth over long videos (FIFO eviction).
MAX_VIDEO_OCR_CACHE_SIZE = 800


def _subtitle_style_scale(style: str) -> float:
    """Return font scale factor for subtitle style (VideoCaptioner-inspired: default, anime, documentary)."""
    if not style:
        return 0.038
    s = style.strip().lower()
    if s == "anime":
        return 0.042   # Slightly larger than default
    if s == "documentary":
        return 0.032   # Compact, more lines fit
    return 0.038       # default: smaller so more text fits on one line


def _compute_ocr_cache_key(blk: TextBlock, img: np.ndarray) -> Optional[Tuple[Any, ...]]:
    """Stable cache key: quantized box geometry + MD5 of downscaled crop so content change invalidates cache."""
    try:
        xyxy = getattr(blk, "xyxy", None)
        if not xyxy or len(xyxy) < 4 or img is None or img.size == 0:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        # Quantize geometry (8px grid) so minor detector jitter doesn't break the cache
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bw = x2 - x1
        bh = y2 - y1
        q = 8
        geo = (cx // q, cy // q, bw // q, bh // q)
        # Small grayscale thumbnail + MD5 for a compact, content-sensitive key
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        thumb = cv2.resize(gray, (16, 4), interpolation=cv2.INTER_AREA)
        content_hash = hashlib.md5(thumb.tobytes()).digest()
        return geo + (content_hash,)
    except Exception:
        return None


def _subtitle_font_paths() -> List[Optional[str]]:
    """Return list of font paths to try for burn-in subtitles (custom first if set, then bold/subtitle-friendly defaults)."""
    paths = []
    custom = (getattr(pcfg.module, "video_translator_subtitle_font", None) or "").strip()
    if custom and os.path.isfile(custom):
        paths.append(custom)
    import sys
    windir = os.environ.get("WINDIR", "C:\\Windows")
    fonts_dir = os.path.join(windir, "Fonts")
    if sys.platform == "win32":
        # Prefer Arial Bold (common for subtitles); then Arial, Tahoma, Verdana
        for name in ("arialbd.ttf", "arial.ttf", "tahomabd.ttf", "tahoma.ttf", "verdanab.ttf", "verdana.ttf"):
            p = os.path.join(fonts_dir, name)
            if p not in paths and os.path.isfile(p):
                paths.append(p)
    for p in (
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if p and p not in paths and os.path.isfile(p):
            paths.append(p)
    return paths


def _subtitle_font_paths_italic() -> List[Optional[str]]:
    """Font paths for italic style (translator/flow fixer can use *italic*)."""
    import sys
    paths: List[Optional[str]] = []
    windir = os.environ.get("WINDIR", "C:\\Windows")
    fonts_dir = os.path.join(windir, "Fonts")
    if sys.platform == "win32":
        for name in ("ariali.ttf", "arialbi.ttf", "tahomabi.ttf", "verdanai.ttf"):
            p = os.path.join(fonts_dir, name)
            if p and os.path.isfile(p):
                paths.append(p)
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
    ):
        if p and os.path.isfile(p):
            paths.append(p)
    return paths


def _subtitle_font_paths_bold_italic() -> List[Optional[str]]:
    """Font paths for bold+italic (***text***)."""
    import sys
    paths: List[Optional[str]] = []
    windir = os.environ.get("WINDIR", "C:\\Windows")
    fonts_dir = os.path.join(windir, "Fonts")
    if sys.platform == "win32":
        for name in ("arialbi.ttf", "tahomabi.ttf", "verdanabi.ttf"):
            p = os.path.join(fonts_dir, name)
            if p and os.path.isfile(p):
                paths.append(p)
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
    ):
        if p and os.path.isfile(p):
            paths.append(p)
    return paths


def _wrap_text_to_width(
    text: str,
    font: Any,
    draw: Any,
    max_px_width: int,
    max_chars: Optional[int] = None,
) -> List[str]:
    """Wrap text into lines that do not exceed max_px_width (and optionally max_chars per line, Netflix-style)."""
    if not text or max_px_width <= 0:
        return [text] if text else []
    text = text.strip()
    if not text:
        return []
    try:
        if hasattr(draw, "textbbox"):
            def measure(s):
                b = draw.textbbox((0, 0), s, font=font)
                return b[2] - b[0]
        else:
            def measure(s):
                b = font.getbbox(s)
                return b[2] - b[0]
    except Exception:
        def measure(s):
            return len(s) * 12
    words = text.split()
    lines = []
    current = []
    current_w = 0
    for word in words:
        w_w = measure(word)
        would_overflow_px = current and current_w + measure(" ") + w_w > max_px_width
        would_overflow_chars = False
        if max_chars and current:
            new_line = " ".join(current) + " " + word
            would_overflow_chars = len(new_line) > max_chars
        if current and (would_overflow_px or would_overflow_chars):
            lines.append(" ".join(current))
            current = [word]
            current_w = w_w
        else:
            current.append(word)
            current_w = current_w + (measure(" ") if current else 0) + w_w
    if current:
        lines.append(" ".join(current))
    if not lines:
        lines = [text]
    return lines


def _parse_subtitle_markup(line: str) -> List[Tuple[str, str]]:
    """Parse *italic*, **bold**, ***bold italic*** into segments. Returns list of (text, style) with style in normal|italic|bold|bold_italic."""
    if not line:
        return [("", "normal")]
    segments: List[Tuple[str, str]] = []
    last_end = 0
    # Match ***...*** first, then **...**, then *...* (non-greedy)
    for m in re.finditer(r"\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*", line):
        if m.start() > last_end:
            segments.append((line[last_end : m.start()], "normal"))
        if m.group(1) is not None:
            segments.append((m.group(1), "bold_italic"))
        elif m.group(2) is not None:
            segments.append((m.group(2), "bold"))
        else:
            segments.append((m.group(3), "italic"))
        last_end = m.end()
    if last_end < len(line):
        segments.append((line[last_end:], "normal"))
    return segments if segments else [("", "normal")]


def _draw_line_segments(
    draw: Any,
    segments: List[Tuple[str, str]],
    fonts: dict,
    px: int,
    py: int,
    outline_color: Tuple[int, int, int],
    fill_color: Tuple[int, int, int],
    outline_radius: int = 2,
) -> int:
    """Draw a line with mixed normal/italic/bold segments; return total width in pixels."""
    total_w = 0
    for text, style in segments:
        if not text:
            continue
        font = fonts.get(style) or fonts.get("normal")
        if not font:
            continue
        try:
            if hasattr(draw, "textbbox"):
                b = draw.textbbox((0, 0), text, font=font)
            else:
                b = font.getbbox(text)
            w = b[2] - b[0]
        except Exception:
            w = len(text) * 12
        for dx in range(-outline_radius, outline_radius + 1):
            for dy in range(-outline_radius, outline_radius + 1):
                if dx != 0 or dy != 0:
                    draw.text((px + dx, py + dy), text, font=font, fill=outline_color)
        draw.text((px, py), text, font=font, fill=fill_color)
        px += w
        total_w += w
    return total_w


def _measure_line_segments(segments: List[Tuple[str, str]], fonts: dict, draw: Any) -> Tuple[int, int]:
    """Return (total_width, max_height) for a line of segments."""
    total_w = 0
    max_h = 0
    for text, style in segments:
        if not text:
            continue
        font = fonts.get(style) or fonts.get("normal")
        if not font:
            continue
        try:
            if hasattr(draw, "textbbox"):
                b = draw.textbbox((0, 0), text, font=font)
            else:
                b = font.getbbox(text)
            total_w += b[2] - b[0]
            max_h = max(max_h, b[3] - b[1])
        except Exception:
            total_w += len(text) * 12
            max_h = max(max_h, 20)
    return total_w, max_h


def _load_subtitle_fonts(size: int) -> dict:
    """Load normal (bold), italic, bold_italic fonts for subtitle markup. Returns dict style -> ImageFont."""
    result = {"normal": None, "italic": None, "bold": None, "bold_italic": None}
    if not HAS_PIL:
        return result
    for path in _subtitle_font_paths():
        if path:
            try:
                result["normal"] = ImageFont.truetype(path, size)
                result["bold"] = result["normal"]
                break
            except Exception:
                pass
    if result["normal"] is None:
        result["normal"] = ImageFont.load_default()
        result["bold"] = result["normal"]
    for path in _subtitle_font_paths_italic():
        if path and result["italic"] is None:
            try:
                result["italic"] = ImageFont.truetype(path, size)
                break
            except Exception:
                pass
    for path in _subtitle_font_paths_bold_italic():
        if path and result["bold_italic"] is None:
            try:
                result["bold_italic"] = ImageFont.truetype(path, size)
                break
            except Exception:
                pass
    if result["italic"] is None:
        result["italic"] = result["normal"]
    if result["bold_italic"] is None:
        result["bold_italic"] = result["italic"]
    return result


def _is_garbage_ocr_source(text: str) -> bool:
    """True if the OCR source looks like noise (e.g. single letter, fragment like 'utive') so we should not burn it in as a subtitle."""
    if not text or not isinstance(text, str):
        return True
    s = text.strip()
    if not s:
        return True
    # Single/double ASCII letter or short ASCII-only fragment (e.g. S, X, The, utive) — likely OCR noise
    if len(s) <= 6 and s.isascii() and s.isalpha():
        return True
    return False


def _draw_text_on_image(
    img: np.ndarray,
    blk_list: List[TextBlock],
    font_scale: float = 0.5,
    thickness: int = 2,
    color: Tuple[int, int, int] = (255, 255, 255),
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    style: str = "default",
) -> None:
    """Draw each block's translation on img in-place. Prefer PIL for Unicode; fallback to cv2.putText.
    style: default | anime | documentary (VideoCaptioner-inspired subtitle look).
    Blocks whose source text is garbage OCR (e.g. single letter, fragment) are skipped and not drawn."""
    if not blk_list:
        return
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Work on a copy for PIL path so we always write back the full drawn result into img
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    scale_factor = _subtitle_style_scale(style)
    font = None
    fonts_styled = {}
    if HAS_PIL:
        # Scale font with resolution and style (research: ~35–55px at 1080p, Netflix-relative sizing)
        size = max(18, min(96, int(min(w, h) * scale_factor)))
        for path in _subtitle_font_paths():
            if path:
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()
        fonts_styled = _load_subtitle_fonts(size)

    # Minimum line height from font so text is never squished (e.g. "Ay" cap height + padding)
    min_line_h = 20
    if font and HAS_PIL:
        try:
            if hasattr(font, "getbbox"):
                bbox = font.getbbox("Ay")
            else:
                bbox = (0, 0, 20, 24)
            min_line_h = max(20, (bbox[3] - bbox[1]) + 6)
        except Exception:
            pass
    draw = ImageDraw.Draw(img_pil)
    # Single style: white fill, black outline (no second color so no double/layered subtitle look)
    color_rgb = (color[2], color[1], color[0]) if len(color) >= 3 else (255, 255, 255)  # BGR -> RGB
    for blk in blk_list:
        raw_src = blk.get_text() if callable(getattr(blk, "get_text", None)) else getattr(blk, "text", "")
        src = (raw_src if isinstance(raw_src, str) else " ".join(str(x or "") for x in (raw_src or []))).strip()
        if _is_garbage_ocr_source(src):
            continue
        trans = (getattr(blk, "translation", None) or "").strip()
        if not trans:
            continue
        use_color = color_rgb
        xyxy = getattr(blk, "xyxy", None)
        if not xyxy or len(xyxy) < 4:
            continue
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        # Split into lines if contains newline, then wrap each line to fit box width
        raw_lines = [ln for ln in trans.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]
        if not raw_lines:
            continue
        box_w = max(20, x2 - x1)
        # Use at least 70% of frame width for wrapping so short phrases stay on one line instead of 2-3
        box_w = max(box_w, int(w * 0.70))
        box_h = y2 - y1
        lines = []
        for ln in raw_lines:
            if font and HAS_PIL:
                lines.extend(_wrap_text_to_width(ln, font, draw, box_w, MAX_SUBTITLE_CHARS_PER_LINE))
            else:
                lines.append(ln)
        if not lines:
            continue
        # Use minimum line height so translation is never squished; allow box to extend down if needed
        line_h = max(min_line_h, box_h // len(lines))
        total_text_h = line_h * len(lines)
        # Extend drawing region downward if text needs more space than original bubble
        y2_draw = min(h, y1 + max(box_h, total_text_h))
        y_start = y1 + max(0, ((y2_draw - y1) - total_text_h) // 2)
        # Keep subtitles above bottom safe area so they are not cut off (overscan, player chrome)
        bottom_safe = int(h * BOTTOM_SAFE_FRAC)
        if y_start + total_text_h > h - bottom_safe:
            y_start = max(0, h - total_text_h - bottom_safe)
        y_cur = y_start
        bottom_limit = h - bottom_safe
        outline_rgb = (outline_color[2], outline_color[1], outline_color[0]) if len(outline_color) >= 3 else (0, 0, 0)
        for line in lines:
            if y_cur + line_h > bottom_limit:
                break
            if font and HAS_PIL:
                segments = _parse_subtitle_markup(line)
                has_markup = any(s[1] != "normal" for s in segments)
                if has_markup and fonts_styled:
                    tw, th = _measure_line_segments(segments, fonts_styled, draw)
                    if tw > x2 - x1:
                        px = max(0, (w - tw) // 2)
                    else:
                        px = x1 + max(0, (x2 - x1 - tw) // 2)
                        px = max(x1, min(x2 - tw, px))
                    py = y_cur + max(0, (line_h - th) // 2)
                    _draw_line_segments(draw, segments, fonts_styled, px, py, outline_rgb, use_color, outline_radius=2)
                else:
                    try:
                        if hasattr(draw, "textbbox"):
                            bbox = draw.textbbox((0, 0), line, font=font)
                        else:
                            bbox = font.getbbox(line)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                    except Exception:
                        tw, th = 50, 20
                    if tw > x2 - x1:
                        px = max(0, (w - tw) // 2)
                    else:
                        px = x1 + max(0, (x2 - x1 - tw) // 2)
                        px = max(x1, min(x2 - tw, px))
                    py = y_cur + max(0, (line_h - th) // 2)
                    for dx in (-2, -1, 0, 1, 2):
                        for dy in (-2, -1, 0, 1, 2):
                            if dx != 0 or dy != 0:
                                draw.text((px + dx, py + dy), line, font=font, fill=outline_rgb)
                    draw.text((px, py), line, font=font, fill=use_color)
            else:
                # cv2 fallback (ASCII only)
                blk_color = color
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                if tw > x2 - x1:
                    px = max(0, (w - tw) // 2)
                else:
                    px = x1 + max(0, (x2 - x1 - tw) // 2)
                py = y_cur + min(th, line_h - 2)
                # Small black outline (2px) for readability on any background
                for dx in (-2, -1, 0, 1, 2):
                    for dy in (-2, -1, 0, 1, 2):
                        if dx != 0 or dy != 0:
                            cv2.putText(img, line, (px + dx, py + dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 2)
                cv2.putText(img, line, (px, py), cv2.FONT_HERSHEY_SIMPLEX, font_scale, blk_color, thickness)
            y_cur += line_h
        # end for line
    # end for blk

    # Write drawn image back into caller's buffer so the video frame definitely contains the text
    if HAS_PIL:
        result_bgr = cv2.cvtColor(np.ascontiguousarray(np.array(img_pil)), cv2.COLOR_RGB2BGR)
        if result_bgr.shape == img.shape:
            np.copyto(img, result_bgr)
        else:
            img[:] = result_bgr[: img.shape[0], : img.shape[1]]


def _draw_timed_subs_on_image(
    img: np.ndarray,
    time_sec: float,
    segments: List[Tuple],
    style: str = "default",
    color: Tuple[int, int, int] = (255, 255, 255),
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    stack_multiple_lines: bool = False,
) -> None:
    """Draw subtitles: segments are (start_sec, end_sec, text) or (start_sec, end_sec, text, style_override). Draw active at bottom center; per-segment style if 4th element present. If stack_multiple_lines is True and multiple segments are active, draw each on its own line (first above, next below) instead of joining."""
    # active: list of (text, style_to_use)
    active = []
    for seg in segments:
        s, e = seg[0], seg[1]
        text = (seg[2] if len(seg) > 2 else "").strip()
        if s <= time_sec < e and text:
            style_override = (seg[3] if len(seg) > 3 and seg[3] else None) or style
            active.append((text, style_override))
    if not active:
        return
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Stack each segment on its own line when requested (e.g. close back-to-back subs)
    if stack_multiple_lines and len(active) > 1:
        y_offset = h - int(h * BOTTOM_SAFE_FRAC)
        style_used = active[0][1]
        for text, style_used in reversed(active):
            scale_factor = _subtitle_style_scale(style_used)
            line_h = _draw_single_style_subtitle(img, text, scale_factor, w, h, color, outline_color, y_offset=y_offset)
            y_offset -= line_h
        return
    # Single style: combine all text (original behavior)
    if len(active) == 1 or all(a[1] == active[0][1] for a in active):
        combined = " ".join(t.strip() for t, _ in active).strip()
        style_used = active[0][1]
        scale_factor = _subtitle_style_scale(style_used)
        _draw_single_style_subtitle(img, combined, scale_factor, w, h, color, outline_color)
        return
    # Per-segment style: draw each active segment with its style (stacked)
    y_offset = h - int(h * BOTTOM_SAFE_FRAC)
    for text, style_used in reversed(active):
        scale_factor = _subtitle_style_scale(style_used)
        line_h = _draw_single_style_subtitle(img, text, scale_factor, w, h, color, outline_color, y_offset=y_offset)
        y_offset -= line_h


def _draw_single_style_subtitle(
    img: np.ndarray,
    combined: str,
    scale_factor: float,
    w: int,
    h: int,
    color: Tuple[int, int, int],
    outline_color: Tuple[int, int, int],
    y_offset: Optional[int] = None,
) -> int:
    """Draw one block of subtitle text; return line height used. If y_offset set, draw at that y (from bottom).
    Text is wrapped to fit within the image width so it never goes off-screen."""
    if not (combined or "").strip():
        return 0
    font_scale = max(0.32, min(1.0, min(w, h) * scale_factor / 400.0))
    thickness = max(1, int(2 * font_scale))
    if not HAS_PIL:
        # cv2 fallback (ASCII-only): wrap by rough char count to fit ~85% width
        raw = [ln for ln in combined.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()] or [combined]
        max_ch = min(MAX_SUBTITLE_CHARS_PER_LINE, max(10, int(w * 0.85 / (12 * font_scale))))
        lines = []
        for ln in raw:
            while len(ln) > max_ch:
                lines.append(ln[:max_ch].rsplit(" ", 1)[0] or ln[:max_ch])
                ln = ln[len(lines[-1]):].lstrip()
            if ln:
                lines.append(ln)
        if not lines:
            lines = [combined]
        line_h = int(30 * font_scale)
        total_h = line_h * len(lines)
        bottom_margin = max(12, int(h * BOTTOM_SAFE_FRAC))
        top_margin = max(4, int(h * 0.02))
        available_h = h - bottom_margin - top_margin
        while total_h > available_h and font_scale > 0.25 and len(lines) > 0:
            font_scale = max(0.25, font_scale * 0.85)
            thickness = max(1, int(2 * font_scale))
            line_h = int(30 * font_scale)
            total_h = line_h * len(lines)
        y_start = (y_offset - total_h) if y_offset is not None else max(top_margin, h - total_h - bottom_margin)
        y_start = max(top_margin, min(y_start, h - total_h - bottom_margin))
        if y_start < top_margin:
            y_start = top_margin
        bottom_limit = h - bottom_margin
        for i, line in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            px = max(0, min(w - tw, (w - tw) // 2))
            py = y_start + i * line_h + th
            if py > bottom_limit:
                break
            # Small black outline (2px) for readability on any background
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    if dx != 0 or dy != 0:
                        cv2.putText(img, line, (px + dx, py + dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 2)
            cv2.putText(img, line, (px, py), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return total_h
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    size = max(18, min(96, int(min(w, h) * scale_factor)))
    font = None
    for path in _subtitle_font_paths():
        if path:
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    max_line_px = max(80, int(w * 0.88))
    raw_lines = [ln for ln in combined.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]
    if not raw_lines:
        raw_lines = [combined]
    lines = []
    for ln in raw_lines:
        lines.extend(_wrap_text_to_width(ln, font, draw, max_line_px, MAX_SUBTITLE_CHARS_PER_LINE))
    if not lines:
        lines = [combined]
    try:
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), lines[0], font=font)
        else:
            bbox = font.getbbox(lines[0])
        line_h = bbox[3] - bbox[1] + 4
    except Exception:
        line_h = 24
    total_h = line_h * len(lines)
    bottom_margin = max(12, int(h * BOTTOM_SAFE_FRAC))
    top_margin = max(4, int(h * 0.02))
    available_h = h - bottom_margin - top_margin
    # Scale down font so multiple lines never go off bottom
    while total_h > available_h and size > 8 and len(lines) > 0:
        size = max(8, int(size * 0.85))
        font = None
        for path in _subtitle_font_paths():
            if path:
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()
        try:
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), lines[0], font=font)
            else:
                bbox = font.getbbox(lines[0])
            line_h = bbox[3] - bbox[1] + 4
        except Exception:
            line_h = max(12, line_h - 2)
        total_h = line_h * len(lines)
    fonts_styled = _load_subtitle_fonts(size)
    color_rgb = (color[2], color[1], color[0]) if len(color) >= 3 else (255, 255, 255)
    outline_rgb = (outline_color[2], outline_color[1], outline_color[0]) if len(outline_color) >= 3 else (0, 0, 0)
    y_start = (y_offset - total_h) if y_offset is not None else max(top_margin, h - total_h - bottom_margin)
    y_start = max(top_margin, min(y_start, h - total_h - bottom_margin))
    if y_start < top_margin:
        y_start = top_margin
    bottom_limit = h - bottom_margin
    y_cur = y_start
    for line in lines:
        if y_cur + line_h > bottom_limit:
            break
        segments = _parse_subtitle_markup(line)
        has_markup = any(s[1] != "normal" for s in segments)
        if has_markup and fonts_styled.get("normal"):
            tw, th = _measure_line_segments(segments, fonts_styled, draw)
            px = max(0, min(w - tw, (w - tw) // 2))
            py = y_cur
            _draw_line_segments(draw, segments, fonts_styled, px, py, outline_rgb, color_rgb, outline_radius=2)
        else:
            try:
                if hasattr(draw, "textbbox"):
                    bbox = draw.textbbox((0, 0), line, font=font)
                else:
                    bbox = font.getbbox(line)
                tw = bbox[2] - bbox[0]
            except Exception:
                tw = 200
            px = max(0, min(w - tw, (w - tw) // 2))
            py = y_cur
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    if dx != 0 or dy != 0:
                        draw.text((px + dx, py + dy), line, font=font, fill=outline_rgb)
            draw.text((px, py), line, font=font, fill=color_rgb)
        y_cur += line_h
    result_bgr = cv2.cvtColor(np.ascontiguousarray(np.array(img_pil)), cv2.COLOR_RGB2BGR)
    if y_offset is not None:
        blit_y = max(0, y_start)
        roi_h = min(total_h, h - blit_y)
        if roi_h > 0 and result_bgr.shape[0] >= blit_y + roi_h:
            img[blit_y : blit_y + roi_h, :] = result_bgr[blit_y : blit_y + roi_h, :]
    else:
        if result_bgr.shape == img.shape:
            np.copyto(img, result_bgr)
        else:
            img[:] = result_bgr[: img.shape[0], : img.shape[1]]
    return total_h


def _region_fraction_from_preset(cfg: Any) -> float:
    """Return fraction of frame height for subtitle region (e.g. 0.2 = bottom 20%). 0 = full frame."""
    preset = (getattr(cfg, "video_translator_region_preset", None) or "full").strip().lower()
    if not preset or preset == "full":
        return 0.0
    if "bottom_30" in preset:
        return 0.30
    if "bottom_25" in preset:
        return 0.25
    if "bottom_20" in preset:
        return 0.20
    if "bottom_15" in preset:
        return 0.15
    return 0.20


def _make_fixed_region_block(im_w: int, im_h: int, frac: float) -> TextBlock:
    """One block covering the bottom frac of the frame (for skip-detect / fixed region)."""
    y_min = int(im_h * (1.0 - frac))
    y_min = max(0, min(y_min, im_h - 1))
    blk = TextBlock()
    blk.xyxy = [0, y_min, im_w, im_h]
    blk.lines = [[[0, y_min], [im_w, y_min], [im_w, im_h], [0, im_h]]]
    return blk


def run_one_frame_pipeline(
    img: np.ndarray,
    detector: Any,
    ocr: Any,
    translator: Any,
    inpainter: Any,
    enable_detect: bool,
    enable_ocr: bool,
    enable_translate: bool,
    enable_inpaint: bool,
    cfg: Any = None,
    skip_detect: bool = False,
    draw_subtitles: bool = True,
) -> Tuple[np.ndarray, List[TextBlock], Optional[np.ndarray]]:
    """
    Run detect -> OCR -> translate -> inpaint on a single frame. Returns (out_img, blk_list, mask).
    If skip_detect and region preset is not full, uses a single fixed block for the subtitle band (no detector).
    When draw_subtitles is False (inpaint-only soft subs), inpainted frame is returned without drawing text.
    """
    cfg = cfg or pcfg.module
    im_h, im_w = img.shape[:2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    blk_list: List[TextBlock] = []
    mask: Optional[np.ndarray] = None
    frac = _region_fraction_from_preset(cfg)

    if skip_detect and frac > 0:
        blk_list = [_make_fixed_region_block(im_w, im_h, frac)]
    elif enable_detect and detector is not None:
        try:
            mask, blk_list = detector.detect(img, None)
            if blk_list:
                for blk in blk_list:
                    if getattr(blk, "lines", None) and len(blk.lines) > 0:
                        examine_textblk(blk, im_w, im_h, sort=True)
                blk_list = remove_contained_boxes(blk_list)
                blk_list = deduplicate_primary_boxes(blk_list, iou_threshold=0.5)
                # Restrict to subtitle region (e.g. bottom 20%) when preset is set — exclude jacket/caption text.
                # Require block center to be inside the bottom band so boxes that only barely overlap are dropped.
                if frac > 0 and im_h > 0:
                    y_min_region = im_h * (1.0 - frac)
                    kept = []
                    for blk in blk_list:
                        xyxy = getattr(blk, "xyxy", None)
                        if not xyxy or len(xyxy) < 4:
                            kept.append(blk)
                            continue
                        y1, y2 = float(xyxy[1]), float(xyxy[3])
                        center_y = (y1 + y2) * 0.5
                        if center_y >= y_min_region and y1 < im_h:
                            kept.append(blk)
                    blk_list = kept
        except Exception:
            blk_list = []

    if not blk_list:
        return img.copy(), [], None

    if enable_ocr and ocr is not None:
        try:
            from modules.ocr.base import ensure_blocks_have_lines, normalize_block_text
            ensure_blocks_have_lines(blk_list)
            # Per-region OCR cache: reuse OCR when geometry + content are unchanged (bounded, FIFO eviction)
            cache = getattr(ocr, "_video_frame_ocr_cache", None)
            cache_order = getattr(ocr, "_video_frame_ocr_cache_order", None)
            if cache is None:
                cache = {}
                cache_order = []
                setattr(ocr, "_video_frame_ocr_cache", cache)
                setattr(ocr, "_video_frame_ocr_cache_order", cache_order)
            need_ocr: List[TextBlock] = []
            pending_keys: List[Optional[Tuple[Any, ...]]] = []
            n_hits = 0
            for blk in blk_list:
                key = _compute_ocr_cache_key(blk, img)
                if key is not None and key in cache:
                    cached_text = cache[key]
                    if isinstance(getattr(blk, "text", None), list):
                        blk.text = [cached_text]
                    else:
                        blk.text = cached_text
                    n_hits += 1
                else:
                    need_ocr.append(blk)
                    pending_keys.append(key)
            if need_ocr:
                ocr.run_ocr(img, need_ocr)
                for blk, key in zip(need_ocr, pending_keys):
                    try:
                        normalize_block_text(blk)
                        txt = blk.get_text() or ""
                        if key is not None and txt.strip():
                            evicted = 0
                            while len(cache) >= MAX_VIDEO_OCR_CACHE_SIZE and cache_order:
                                old_key = cache_order.pop(0)
                                cache.pop(old_key, None)
                                evicted += 1
                            if evicted:
                                LOGGER.debug(
                                    "Video OCR cache: evicted %d entries (max %d)",
                                    evicted, MAX_VIDEO_OCR_CACHE_SIZE,
                                )
                            if key not in cache:
                                cache_order.append(key)
                            cache[key] = txt
                    except Exception:
                        continue
            n_misses = len(need_ocr)
            # Only log when we have both hits and misses (indicates cache is actively helping),
            # to avoid spamming logs when building the cache for the first time.
            if n_hits and n_misses:
                LOGGER.debug(
                    "Video OCR cache: %d hit(s), %d miss(es), ran OCR for %d block(s), cache size %d",
                    n_hits, n_misses, n_misses, len(cache),
                )
        except Exception:
            pass

    # Optional LLM OCR correction (VideoCaptioner-style) before translation
    if translator is not None and getattr(translator, "get_param_value", None) and translator.get_param_value("correct_ocr_with_llm") and blk_list:
        try:
            texts = [blk.get_text() or "" for blk in blk_list]
            if any(t.strip() for t in texts):
                corrected = translator.correct_ocr_texts(texts)
                if corrected and len(corrected) == len(blk_list):
                    for blk, new_text in zip(blk_list, corrected):
                        blk.text = [new_text] if isinstance(getattr(blk, "text", None), list) else new_text
                        if not getattr(blk, "lines", None):
                            blk.lines = []
        except Exception:
            pass

    if enable_translate and translator is not None:
        try:
            setattr(translator, "_current_page_key", "video_frame")
            setattr(translator, "_current_page_image", img)
            # Simple per-run cache so identical source text across frames reuses
            # the same translations instead of calling the LLM again.
            texts = [blk.get_text() or "" for blk in blk_list]
            key = tuple(texts)
            cache = getattr(translator, "_video_frame_cache", None)
            if cache is None:
                cache = {}
                setattr(translator, "_video_frame_cache", cache)
            cached = cache.get(key)
            if cached and len(cached) == len(blk_list):
                for blk, trans in zip(blk_list, cached):
                    blk.translation = trans
            else:
                prev_cached = cached if (cached and isinstance(cached, list)) else None
                translator.translate_textblk_lst(blk_list)
                # Guard against LLM glitches: if the model returns a count mismatch or blanks for some
                # items, do NOT overwrite a previously good cached translation with empty strings.
                new_list = []
                for i, blk in enumerate(blk_list):
                    t = str(getattr(blk, "translation", "") or "").strip()
                    if (not t) and prev_cached and i < len(prev_cached):
                        prev_t = str(prev_cached[i] or "").strip()
                        if prev_t:
                            t = prev_t
                            blk.translation = t
                    if not t:
                        # Last resort: keep something stable so UI/compositing doesn't flicker off.
                        t = (blk.get_text() or "").strip()
                        blk.translation = t
                    new_list.append(t)
                cache[key] = new_list
                try:
                    msg = "; ".join(
                        (blk.get_text() or "").strip() + " -> " + str(blk.translation or "").replace("\n", " ").strip()
                        for blk in blk_list
                    )
                except Exception:
                    msg = ""
                if msg:
                    LOGGER.info(
                        "Video translator: translated %d blocks: %s",
                        len(blk_list),
                        msg,
                    )
                else:
                    LOGGER.info("Video translator: translated %d blocks.", len(blk_list))
        except Exception:
            for blk in blk_list:
                if getattr(blk, "translation", None) is None or str(blk.translation).strip() == "":
                    blk.translation = blk.get_text() or ""

    if enable_inpaint and inpainter is not None and blk_list:
        try:
            # Video inpaint caching: only run inpaint when subtitle content/region changes.
            # This matches the "OCR cache" idea and avoids redundant inpaint calls when the same subtitle
            # persists across multiple pipeline ticks (e.g. Process every 30 frames).
            #
            # Note: This intentionally reuses the last inpainted frame result when the subtitle is unchanged,
            # which is the behavior the video pipeline already leans on via cached-result compositing.
            try:
                # Prefer a content-sensitive key: quantized geometry + tiny crop hash per block.
                keys = []
                for b in blk_list:
                    k = _compute_ocr_cache_key(b, img)
                    if k is not None:
                        keys.append(k)
                if keys:
                    inpaint_key = ("v1", tuple(keys))
                else:
                    texts = tuple((b.get_text() or "").strip() for b in blk_list)
                    inpaint_key = ("v1_text", texts)
            except Exception:
                inpaint_key = None

            if inpaint_key is not None:
                last_key = getattr(inpainter, "_video_frame_inpaint_cache_key", None)
                last_img = getattr(inpainter, "_video_frame_inpaint_cache_img", None)
                last_mask = getattr(inpainter, "_video_frame_inpaint_cache_mask", None)
                if last_key == inpaint_key and last_img is not None:
                    # Reuse cached inpainted REGIONS (not the whole frame) for unchanged subtitles.
                    # Reusing the whole cached frame can visually "freeze" motion and look like low FPS.
                    out = img.copy()
                    try:
                        if out is not None and out.shape == last_img.shape and blk_list:
                            h2, w2 = out.shape[:2]
                            for blk in blk_list:
                                xyxy = getattr(blk, "xyxy", None)
                                if not xyxy or len(xyxy) < 4:
                                    continue
                                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                                x1, x2 = max(0, min(x1, w2)), max(0, min(x2, w2))
                                y1, y2 = max(0, min(y1, h2)), max(0, min(y2, h2))
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                out[y1:y2, x1:x2] = last_img[y1:y2, x1:x2]
                        else:
                            # Fallback: if shapes mismatch, just keep current frame (skip inpaint).
                            out = img.copy()
                    except Exception:
                        out = img.copy()
                    return out, blk_list, last_mask

            from modules.textdetector.outside_text_processor import OSB_LABELS
            text_for_nudge = [b for b in blk_list if (getattr(b, "label", None) or "").strip().lower() in OSB_LABELS]
            mask = build_mask_with_resolved_overlaps(
                blk_list, im_w, im_h,
                text_blocks_for_nudge=text_for_nudge if text_for_nudge else None,
            )
            if mask is not None and mask.size > 0:
                # Slight dilation so inpainting covers edges and no "corners" of the box show through
                dilate_px = max(2, min(8, min(im_w, im_h) // 200))
                kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
                mask = cv2.dilate((mask > 127).astype(np.uint8), kernel)
                mask = (mask > 0).astype(np.uint8) * 255
                blk_list_arg = None if getattr(cfg, "inpaint_full_image", False) else blk_list
                img = inpainter.inpaint(img, mask, blk_list_arg)
                # Update cache only when we actually inpainted.
                if inpaint_key is not None and img is not None:
                    try:
                        setattr(inpainter, "_video_frame_inpaint_cache_key", inpaint_key)
                        setattr(inpainter, "_video_frame_inpaint_cache_img", img.copy() if hasattr(img, "copy") else img)
                        setattr(inpainter, "_video_frame_inpaint_cache_mask", mask.copy() if hasattr(mask, "copy") else mask)
                    except Exception:
                        pass
        except Exception:
            pass

    out = img.copy() if img is not None else np.zeros((im_h, im_w, 3), dtype=np.uint8)
    if draw_subtitles and blk_list:
        style = (getattr(cfg, "video_translator_subtitle_style", None) or "default").strip().lower()
        if style not in ("anime", "documentary"):
            style = "default"
        _draw_text_on_image(out, blk_list, style=style)
    return out, blk_list, mask
