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
import time
import numpy as np
import cv2
from typing import List, Optional, Tuple, Any, Dict

from utils.textblock import TextBlock, examine_textblk, remove_contained_boxes, deduplicate_primary_boxes
from utils.config import pcfg
from modules.inpaint.base import build_mask_with_resolved_overlaps
from utils.logger import logger as LOGGER
from modules.flow_fixer.corrections import (
    correct_ocr_via_fixer,
    correct_asr_via_fixer,
    reflect_translations_via_fixer,
)

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

# Model/OCR template junk (e.g. LM Studio or bad OCR returns "1. Line 1 content").
_SUBTITLE_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:\d+\.\s*)?(?:line\s*\d+\s+content|line\s+content|content\s+line\s*\d+)\s*$",
    re.IGNORECASE,
)


def _is_locked_watermark_line(text: str, cfg: Any) -> bool:
    """True when a line matches the user-defined watermark regex lock pattern."""
    if not bool(getattr(cfg, "video_translator_lock_watermark_lines", False)):
        return False
    pattern = str(getattr(cfg, "video_translator_lock_watermark_regex", "") or "").strip()
    if not pattern:
        return False
    t = (text or "").strip()
    if not t:
        return False
    try:
        return bool(re.search(pattern, t))
    except Exception:
        return False


def configure_translator_video_nlp_parallel(translator: Any, cfg: Any = None) -> None:
    """
    Set translator attributes consumed by LLM_API_Translator._translate for VideoCaptioner-style
    chunked / parallel subtitle NLP (video_* and subtitle_file page keys; see video_translator_nlp_* config).
    """
    if translator is None:
        return
    cfg = cfg or pcfg.module
    try:
        chunk = max(0, int(getattr(cfg, "video_translator_nlp_chunk_size", 32) or 0))
    except (TypeError, ValueError):
        chunk = 0
    try:
        workers = max(1, int(getattr(cfg, "video_translator_nlp_max_workers", 1) or 1))
    except (TypeError, ValueError):
        workers = 1
    setattr(translator, "_video_nlp_chunk_size", chunk)
    setattr(translator, "_video_nlp_max_workers", workers)


def clear_translator_video_nlp_parallel(translator: Any) -> None:
    """Remove video NLP parallel attrs so manga/comic translation is unaffected."""
    if translator is None:
        return
    for attr in ("_video_nlp_chunk_size", "_video_nlp_max_workers"):
        try:
            delattr(translator, attr)
        except Exception:
            pass

# Skip-detect fixed-region inpaint mask cache (saves mask-building/dilation work when region preset is constant).
_FIXED_REGION_INPAINT_MASK_CACHE = {}
_FIXED_REGION_INPAINT_MASK_CACHE_ORDER = []
_FIXED_REGION_INPAINT_MASK_CACHE_MAX = 8

# When frame longer side exceeds this, run detector on downscaled frame then scale boxes back (faster, same OCR quality).
_DETECT_MAX_SIDE = 1920


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


def _is_garbage_source(text: str) -> bool:
    """True if source looks like OCR/detection garbage (fragment, single letter, digits only)."""
    t = (text or "").strip()
    if not t:
        return False
    if _SUBTITLE_PLACEHOLDER_RE.match(t):
        return True
    if len(t) > 20:
        return False
    # Single ASCII letter or digit
    if len(t) <= 1:
        return True
    # Digits only (e.g. "11" mistaken as source)
    if t.isdigit():
        return True
    # Short Latin fragment (e.g. "utive" from "executive", "X", "V")
    if len(t) <= 4 and all(ord(c) < 128 for c in t):
        return True
    # Known garbage fragments
    if t.lower() in ("utive", "x", "v", "s"):
        return True
    # Synthetic/test placeholders often returned by noisy OCR/model retries
    tl = t.lower()
    if tl in (
        "line a",
        "line one",
        "text a",
        "text",
        "line",
        "hello",
        "hello world",
        "corrected_text",
        "corrected text",
    ):
        return True
    if tl.startswith(("line ", "text ", "subtitle ")) and len(tl) <= 24:
        return True
    # "line1content" / "lineonecontent" style (no spaces)
    if re.match(r"^line\s*\d*\s*content$", tl):
        return True
    # Punctuation / symbol / emoji-only fragments (e.g. "...", "🌈", "--")
    if all((not ch.isalnum()) and (not ("\u4e00" <= ch <= "\u9fff")) for ch in t):
        return True
    # Repeated single-char junk (e.g. "....", "~~~~", "ーーー")
    if len(set(t)) == 1 and len(t) <= 8:
        return True
    return False


def _normalize_ocr_text_for_compare(text: str) -> str:
    """Normalize OCR text for temporal similarity checks (ignore punctuation/spacing jitter)."""
    t = (text or "").strip().lower()
    if not t:
        return ""
    # Keep letters/numbers/CJK; drop punctuation and whitespace noise.
    return "".join(ch for ch in t if ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"))


def _digit_signature(text: str) -> str:
    """Extract all digits in order; used to protect numeric IDs from bad temporal overrides."""
    return "".join(ch for ch in (text or "") if ch.isdigit())


def _set_block_text(blk: TextBlock, txt: str) -> None:
    if isinstance(getattr(blk, "text", None), list):
        blk.text = [txt]
    else:
        blk.text = txt


def _apply_temporal_ocr_stabilization(
    translator: Any,
    active_blks: List[TextBlock],
    cfg: Any,
    frame_index: Optional[int],
) -> None:
    """
    Reduce OCR per-frame character jitter by voting over recent frames for the same region.
    """
    enabled = bool(getattr(cfg, "video_translator_ocr_temporal_stability", True))
    if (not enabled) or (not active_blks):
        return
    try:
        window = max(2, min(12, int(getattr(cfg, "video_translator_ocr_temporal_window", 5) or 5)))
        min_votes = max(2, min(window, int(getattr(cfg, "video_translator_ocr_temporal_min_votes", 2) or 2)))
        quant = max(4, min(64, int(getattr(cfg, "video_translator_ocr_temporal_geo_quantization", 24) or 24)))
    except (TypeError, ValueError):
        window, min_votes, quant = 5, 2, 24

    hist: Dict[Tuple[int, int, int, int, int], List[str]] = getattr(
        translator, "_video_ocr_temporal_history", None
    )
    if hist is None:
        hist = {}
        setattr(translator, "_video_ocr_temporal_history", hist)

    max_keys = 256
    for i, blk in enumerate(active_blks):
        src = (blk.get_text() or "").strip()
        if not src or _is_garbage_source(src):
            continue
        xyxy = getattr(blk, "xyxy", None) or [0, 0, 0, 0]
        if len(xyxy) < 4:
            continue
        key = (
            i,
            int(round(float(xyxy[0]) / quant)),
            int(round(float(xyxy[1]) / quant)),
            int(round(float(xyxy[2]) / quant)),
            int(round(float(xyxy[3]) / quant)),
        )
        seq = hist.get(key)
        if seq is None:
            seq = []
            hist[key] = seq
        seq.append(src)
        if len(seq) > window:
            del seq[:-window]
        if len(hist) > max_keys:
            # Drop oldest inserted key (dict preserves insertion order in py3.7+)
            try:
                hist.pop(next(iter(hist)))
            except Exception:
                pass

        # Vote on normalized forms; choose representative with highest count.
        norm_counts: Dict[str, int] = {}
        rep_by_norm: Dict[str, str] = {}
        for s in seq:
            n = _normalize_ocr_text_for_compare(s)
            if not n:
                continue
            norm_counts[n] = norm_counts.get(n, 0) + 1
            # Keep the longest representative string for that normalized form.
            prev = rep_by_norm.get(n, "")
            if len((s or "").strip()) >= len((prev or "").strip()):
                rep_by_norm[n] = s
        if not norm_counts:
            continue
        best_norm, best_cnt = max(norm_counts.items(), key=lambda kv: kv[1])
        if best_cnt >= min_votes:
            stable = (rep_by_norm.get(best_norm) or "").strip()
            if stable and stable != src:
                # Guard: do not override numeric-heavy lines unless digits stay identical.
                src_digits = _digit_signature(src)
                stable_digits = _digit_signature(stable)
                if (
                    len(src_digits) >= 6
                    and len(stable_digits) >= 6
                    and src_digits != stable_digits
                ):
                    continue
                _set_block_text(blk, stable)
                LOGGER.debug(
                    "Video OCR stabilize: frame=%s idx=%d votes=%d/%d '%s' -> '%s'",
                    str(frame_index) if frame_index is not None else "?",
                    i,
                    best_cnt,
                    len(seq),
                    src,
                    stable,
                )


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
        # Quantize geometry so minor detector jitter doesn't break the cache.
        # Configurable to tune hit-rate vs reuse-risk.
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bw = x2 - x1
        bh = y2 - y1
        q = int(getattr(pcfg.module, "video_translator_ocr_cache_geo_quantization", 8) or 8)
        q = max(2, min(32, q))
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


def _sanitize_blocks_for_ocr(blk_list: List[TextBlock], im_w: int, im_h: int) -> None:
    """
    Normalize detector output before OCR.
    If polygon lines are malformed or suspiciously larger than bbox, fallback to bbox rectangle line.
    """
    for blk in blk_list or []:
        xyxy = getattr(blk, "xyxy", None)
        if not xyxy or len(xyxy) < 4:
            continue
        try:
            x1, y1, x2, y2 = int(float(xyxy[0])), int(float(xyxy[1])), int(float(xyxy[2])), int(float(xyxy[3]))
        except Exception:
            continue
        x1, x2 = max(0, min(x1, im_w)), max(0, min(x2, im_w))
        y1, y2 = max(0, min(y1, im_h)), max(0, min(y2, im_h))
        if x2 <= x1 or y2 <= y1:
            continue
        blk.xyxy = [x1, y1, x2, y2]
        rect_line = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        rect_area = float((x2 - x1) * (y2 - y1))
        lines = getattr(blk, "lines", None)
        if not lines:
            blk.lines = [rect_line]
            continue
        cleaned = []
        suspicious = False
        for line in lines:
            try:
                pts = np.asarray(line, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
                    suspicious = True
                    continue
                pts[:, 0] = np.clip(pts[:, 0], 0, im_w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, im_h - 1)
                area = abs(float(cv2.contourArea(pts.astype(np.float32))))
                # If line polygon area is way larger than bbox area, it is likely malformed.
                if rect_area > 0 and area > (rect_area * 3.0):
                    suspicious = True
                    continue
                cleaned.append(pts.astype(np.int32).tolist())
            except Exception:
                suspicious = True
                continue
        if (not cleaned) or suspicious:
            blk.lines = [rect_line]
        else:
            blk.lines = cleaned


def _compute_skip_detect_band_thumb(
    img: np.ndarray,
    frac: float,
    thumb_w: int = 96,
    thumb_h: int = 24,
) -> Optional[np.ndarray]:
    """Return a small grayscale thumbnail of the bottom subtitle band for frame-diff gating."""
    try:
        if img is None or img.size == 0 or frac <= 0:
            return None
        h, w = img.shape[:2]
        y1 = int(h * (1.0 - frac))
        y1 = max(0, min(y1, h - 1))
        band = img[y1:h, 0:w]
        if band is None or band.size == 0:
            return None
        if band.ndim == 3:
            gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        else:
            gray = band
        tw = max(16, min(256, int(thumb_w)))
        th = max(8, min(128, int(thumb_h)))
        return cv2.resize(gray, (tw, th), interpolation=cv2.INTER_AREA)
    except Exception:
        return None


def _compute_block_thumb(
    img: np.ndarray,
    blk: TextBlock,
    thumb_w: int = 48,
    thumb_h: int = 16,
) -> Optional[np.ndarray]:
    """Small grayscale thumbnail for one block, used by temporal OCR reuse."""
    try:
        if img is None or img.size == 0:
            return None
        xyxy = getattr(blk, "xyxy", None)
        if not xyxy or len(xyxy) < 4:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        tw = max(16, min(128, int(thumb_w)))
        th = max(8, min(64, int(thumb_h)))
        return cv2.resize(gray, (tw, th), interpolation=cv2.INTER_AREA)
    except Exception:
        return None


def _fallback_xyxy_mask(blk_list: List[TextBlock], im_w: int, im_h: int) -> np.ndarray:
    """Conservative rectangle mask from xyxy only; avoids malformed polygon masks."""
    mask = np.zeros((im_h, im_w), dtype=np.uint8)
    for blk in blk_list or []:
        xyxy = getattr(blk, "xyxy", None)
        if not xyxy or len(xyxy) < 4:
            continue
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        x1, x2 = max(0, min(x1, im_w)), max(0, min(x2, im_w))
        y1, y2 = max(0, min(y1, im_h)), max(0, min(y2, im_h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def _video_subtitle_inpaint_mask_expand(
    mask: np.ndarray,
    im_h: int,
    im_w: int,
    base_dilate_px: int,
) -> np.ndarray:
    """
    Subtitle masks from detectors are often tight on glyph bounds. Anti-aliased edges,
    outlines, and descenders (g, y, j, commas) extend outside the polygon — the inpaint
    region then covers only the top/middle of strokes while the bottom still shows
    original pixels → “malformed” or half-erased text. Expand mostly vertically and
    nudge downward slightly to cover typical subtitle rendering.
    """
    if mask is None or mask.size == 0:
        return mask
    m = (mask > 127).astype(np.uint8)
    if not np.any(m):
        return mask
    k0 = max(1, int(base_dilate_px))
    # Tall rectangular kernel: horizontal text needs more Y coverage than X
    v_h = max(5, min(33, k0 * 2 + 9))
    v_w = max(3, min(25, k0 * 2 + 5))
    if v_h % 2 == 0:
        v_h += 1
    if v_w % 2 == 0:
        v_w += 1
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (v_w, v_h))
    m2 = cv2.dilate(m, kv, iterations=1)
    # OR with mask shifted down a few pixels (bottom bbox edge often clips tails)
    shift = max(1, min(6, im_h // 280))
    m_down = np.zeros_like(m2)
    m_down[shift:, :] = m2[:-shift, :]
    m2 = np.maximum(m2, m_down)
    return ((m2 > 0).astype(np.uint8) * 255).astype(np.uint8)


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
    sl = s.lower()
    # Common synthetic placeholders that should never be burned in as subtitles
    if sl in (
        "line a",
        "line one",
        "text a",
        "text",
        "line",
        "hello",
        "hello world",
        "corrected_text",
        "corrected text",
    ):
        return True
    if sl.startswith(("line ", "text ", "subtitle ")) and len(sl) <= 24:
        return True
    if _SUBTITLE_PLACEHOLDER_RE.match(s):
        return True
    if re.match(r"^line\s*\d*\s*content$", sl):
        return True
    if all((not ch.isalnum()) and (not ("\u4e00" <= ch <= "\u9fff")) for ch in s):
        return True
    if len(set(s)) == 1 and len(s) <= 8:
        return True
    return False


def _block_plain_ocr_text(blk: TextBlock) -> str:
    """Single string source text for a block (same convention as subtitle draw path)."""
    raw_src = blk.get_text() if callable(getattr(blk, "get_text", None)) else getattr(blk, "text", "")
    if isinstance(raw_src, str):
        return raw_src.strip()
    return " ".join(str(x or "") for x in (raw_src or [])).strip()


def _is_substantive_ocr_for_video_inpaint(text: str, detector_block_count: int = 1) -> bool:
    """
    True if OCR output is worth inpainting for (real subtitle to remove).
    Drops empty/garbage. Short digit-only strings are dropped only when detector_block_count > 1
    (likely shard noise); a single full-line box may legitimately be "42" or a countdown digit.
    """
    if not text or not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if _is_garbage_ocr_source(s):
        return False
    if detector_block_count > 1 and re.fullmatch(r"\d{1,2}", s):
        return False
    return True


def _has_drawable_subtitles(blk_list: List[TextBlock]) -> bool:
    """Return True when at least one non-garbage source block has non-empty translation."""
    for blk in blk_list or []:
        raw_src = blk.get_text() if callable(getattr(blk, "get_text", None)) else getattr(blk, "text", "")
        src = (raw_src if isinstance(raw_src, str) else " ".join(str(x or "") for x in (raw_src or []))).strip()
        if _is_garbage_ocr_source(src):
            continue
        trans = (getattr(blk, "translation", None) or "").strip()
        if trans:
            return True
    return False


def _sanitize_subtitle_translation(src: str, trans: str) -> str:
    """Drop obvious meta/test/junk responses so they are not cached or rendered."""
    t = (trans or "").strip()
    if not t:
        return ""
    # Strip trivial angle wrappers: <text> -> text
    if len(t) >= 3 and t.startswith("<") and t.endswith(">"):
        inner = t[1:-1].strip()
        if inner:
            t = inner
    if _SUBTITLE_PLACEHOLDER_RE.match(t):
        return ""
    tl = t.lower()
    # Meta responses from reviewer/corrector style prompts
    if "no source text or current translation provided for review" in tl:
        return ""
    if tl in (
        "line a",
        "line one",
        "text a",
        "text",
        "line",
        "hello",
        "hello world",
        "corrected_text",
        "corrected text",
        "...",
    ):
        return ""
    # Model echoed numbered template as translation (e.g. source and trans both "1. Line 1 content")
    if re.match(r"^\d+\.\s*line\s*\d+\s+content\s*$", tl):
        return ""
    # If source itself is garbage, never keep translation as on-screen subtitle.
    if _is_garbage_source(src):
        return ""
    # Useless echo: translation identical to obvious placeholder source
    if _SUBTITLE_PLACEHOLDER_RE.match((src or "").strip()) and t.strip() == (src or "").strip():
        return ""
    return t


def _subtitle_block_sort_key(b: TextBlock) -> Tuple:
    """Sort key: top-to-bottom, left-to-right; larger area later when overlapping (drawn on top)."""
    xyxy = getattr(b, "xyxy", None)
    if not xyxy or len(xyxy) < 4:
        return (1e18, 1e18, 0, 0)
    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    return (y1, x1, -area, x2 - x1)


def _compute_subtitle_block_draw_bbox(
    blk: TextBlock,
    w: int,
    h: int,
    font: Any,
    draw: Any,
    fonts_styled: dict,
    style: str,
    min_line_h: int,
    font_scale: float,
    thickness: int,
    outline_margin: int = 3,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Union bbox in image coordinates for one block's translation after the same wrap/layout as burn-in.
    Includes a small margin for outline/stroke. Returns None if block is not drawable.
    """
    raw_src = blk.get_text() if callable(getattr(blk, "get_text", None)) else getattr(blk, "text", "")
    src = (raw_src if isinstance(raw_src, str) else " ".join(str(x or "") for x in (raw_src or []))).strip()
    if _is_garbage_ocr_source(src):
        return None
    trans = (getattr(blk, "translation", None) or "").strip()
    if not trans:
        return None
    xyxy = getattr(blk, "xyxy", None)
    if not xyxy or len(xyxy) < 4:
        return None
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    raw_lines = [ln for ln in trans.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]
    if not raw_lines:
        return None
    box_w = max(20, x2 - x1)
    box_w = max(box_w, int(w * 0.70))
    box_h = y2 - y1
    lines: List[str] = []
    for ln in raw_lines:
        if font and HAS_PIL and draw is not None:
            lines.extend(_wrap_text_to_width(ln, font, draw, box_w, MAX_SUBTITLE_CHARS_PER_LINE))
        else:
            lines.append(ln)
    if not lines:
        return None
    line_h = max(min_line_h, box_h // len(lines))
    total_text_h = line_h * len(lines)
    y2_draw = min(h, y1 + max(box_h, total_text_h))
    y_start = y1 + max(0, ((y2_draw - y1) - total_text_h) // 2)
    bottom_safe = int(h * BOTTOM_SAFE_FRAC)
    if y_start + total_text_h > h - bottom_safe:
        y_start = max(0, h - total_text_h - bottom_safe)
    y_cur = y_start
    bottom_limit = h - bottom_safe
    min_x, min_y = 10**9, 10**9
    max_x, max_y = -1, -1
    om = max(0, int(outline_margin))
    for line in lines:
        if y_cur + line_h > bottom_limit:
            break
        if font and HAS_PIL and draw is not None:
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
            min_x = min(min_x, px - om)
            min_y = min(min_y, py - om)
            max_x = max(max_x, px + tw + om)
            max_y = max(max_y, py + th + om)
        else:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            if tw > x2 - x1:
                px = max(0, (w - tw) // 2)
            else:
                px = x1 + max(0, (x2 - x1 - tw) // 2)
            py = y_cur + min(th, line_h - 2)
            min_x = min(min_x, px - om)
            min_y = min(min_y, py - th - om)
            max_x = max(max_x, px + tw + om)
            max_y = max(max_y, py + om)
        y_cur += line_h
    if max_x < min_x or max_y < min_y:
        return None
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(w, max_x)
    max_y = min(h, max_y)
    if max_x <= min_x or max_y <= min_y:
        return None
    return (min_x, min_y, max_x, max_y)


def _draw_text_on_image(
    img: np.ndarray,
    blk_list: List[TextBlock],
    font_scale: float = 0.5,
    thickness: int = 2,
    color: Tuple[int, int, int] = (255, 255, 255),
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    style: str = "default",
    black_box_behind_text: bool = False,
    black_box_padding: int = 6,
    black_box_bgr: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw each block's translation on img in-place. Prefer PIL for Unicode; fallback to cv2.putText.
    style: default | anime | documentary (VideoCaptioner-inspired subtitle look).
    Blocks whose source text is garbage OCR (e.g. single letter, fragment) are skipped and not drawn."""
    if not blk_list:
        return
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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

    ordered_blks = sorted(blk_list, key=_subtitle_block_sort_key)
    # Solid boxes behind subtitles (BGR) before PIL text; uses same wrap/layout as drawing.
    if black_box_behind_text:
        draw_measure = None
        if HAS_PIL:
            measure_img = Image.new("RGB", (max(1, w), max(1, h)), (255, 255, 255))
            draw_measure = ImageDraw.Draw(measure_img)
        pad = max(0, min(64, int(black_box_padding)))
        bb = (
            int(black_box_bgr[0]) if len(black_box_bgr) > 0 else 0,
            int(black_box_bgr[1]) if len(black_box_bgr) > 1 else 0,
            int(black_box_bgr[2]) if len(black_box_bgr) > 2 else 0,
        )
        for blk in ordered_blks:
            bbox_u = _compute_subtitle_block_draw_bbox(
                blk,
                w,
                h,
                font if HAS_PIL else None,
                draw_measure,
                fonts_styled if HAS_PIL else {},
                style,
                min_line_h,
                font_scale,
                thickness,
            )
            if bbox_u is None:
                continue
            x1b, y1b, x2b, y2b = bbox_u
            x1b = max(0, x1b - pad)
            y1b = max(0, y1b - pad)
            x2b = min(w, x2b + pad)
            y2b = min(h, y2b + pad)
            if x2b > x1b and y2b > y1b:
                cv2.rectangle(img, (x1b, y1b), (x2b, y2b), bb, thickness=-1)

    # Work on a copy for PIL path so we always write back the full drawn result into img
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    # Single style: white fill, black outline (no second color so no double/layered subtitle look)
    color_rgb = (color[2], color[1], color[0]) if len(color) >= 3 else (255, 255, 255)  # BGR -> RGB
    for blk in ordered_blks:
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


def subtitle_black_box_draw_kwargs_from_cfg(cfg: Any) -> Dict[str, Any]:
    """Kwargs for _draw_timed_subs_on_image / _draw_single_style_subtitle when using config-driven bar mode."""
    bb_mode = bool(getattr(cfg, "video_translator_subtitle_black_box_mode", False))
    try:
        bb_pad = int(getattr(cfg, "video_translator_subtitle_black_box_padding", 6) or 0)
    except (TypeError, ValueError):
        bb_pad = 6
    bb_pad = max(0, min(64, bb_pad))
    bb_bgr = (
        int(getattr(cfg, "video_translator_subtitle_black_box_b", 0)),
        int(getattr(cfg, "video_translator_subtitle_black_box_g", 0)),
        int(getattr(cfg, "video_translator_subtitle_black_box_r", 0)),
    )
    return {
        "black_box_behind_text": bb_mode,
        "black_box_padding": bb_pad,
        "black_box_bgr": bb_bgr,
    }


def _draw_timed_subs_on_image(
    img: np.ndarray,
    time_sec: float,
    segments: List[Tuple],
    style: str = "default",
    color: Tuple[int, int, int] = (255, 255, 255),
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    stack_multiple_lines: bool = False,
    black_box_behind_text: bool = False,
    black_box_padding: int = 6,
    black_box_bgr: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw subtitles: segments are (start_sec, end_sec, text) or (start_sec, end_sec, text, style_override). Draw active at bottom center; per-segment style if 4th element present. If stack_multiple_lines is True and multiple segments are active, draw each on its own line (first above, next below) instead of joining.
    When black_box_behind_text is True, draws a filled BGR rectangle behind the wrapped text (timed burn-in path)."""
    bb_kw = dict(
        black_box_behind_text=black_box_behind_text,
        black_box_padding=black_box_padding,
        black_box_bgr=black_box_bgr,
    )
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
            line_h = _draw_single_style_subtitle(
                img, text, scale_factor, w, h, color, outline_color, y_offset=y_offset, **bb_kw
            )
            y_offset -= line_h
        return
    # Single style: combine all text (original behavior)
    if len(active) == 1 or all(a[1] == active[0][1] for a in active):
        combined = " ".join(t.strip() for t, _ in active).strip()
        style_used = active[0][1]
        scale_factor = _subtitle_style_scale(style_used)
        _draw_single_style_subtitle(img, combined, scale_factor, w, h, color, outline_color, **bb_kw)
        return
    # Per-segment style: draw each active segment with its style (stacked)
    y_offset = h - int(h * BOTTOM_SAFE_FRAC)
    for text, style_used in reversed(active):
        scale_factor = _subtitle_style_scale(style_used)
        line_h = _draw_single_style_subtitle(
            img, text, scale_factor, w, h, color, outline_color, y_offset=y_offset, **bb_kw
        )
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
    black_box_behind_text: bool = False,
    black_box_padding: int = 6,
    black_box_bgr: Tuple[int, int, int] = (0, 0, 0),
) -> int:
    """Draw one block of subtitle text; return line height used. If y_offset set, draw at that y (from bottom).
    Text is wrapped to fit within the image width so it never goes off-screen.
    If black_box_behind_text, draws a filled rectangle behind all wrapped lines (same geometry as text)."""
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
        if black_box_behind_text and lines:
            om = 3
            pad = max(0, min(64, int(black_box_padding)))
            min_x, min_y = 10**9, 10**9
            max_x, max_y = -1, -1
            for i, line in enumerate(lines):
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                px = max(0, min(w - tw, (w - tw) // 2))
                py = y_start + i * line_h + th
                if py > bottom_limit:
                    break
                min_x = min(min_x, px - om)
                min_y = min(min_y, py - th - om)
                max_x = max(max_x, px + tw + om)
                max_y = max(max_y, py + om)
            if max_x >= min_x and max_y >= min_y:
                x1b = max(0, min_x - pad)
                y1b = max(0, min_y - pad)
                x2b = min(w - 1, max_x + pad)
                y2b = min(h - 1, max_y + pad)
                cv2.rectangle(img, (x1b, y1b), (x2b, y2b), black_box_bgr, thickness=-1)
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
    if black_box_behind_text and lines:
        om = 3
        pad = max(0, min(64, int(black_box_padding)))
        fill_rgb = (
            int(black_box_bgr[2]) if len(black_box_bgr) > 2 else 0,
            int(black_box_bgr[1]) if len(black_box_bgr) > 1 else 0,
            int(black_box_bgr[0]) if len(black_box_bgr) > 0 else 0,
        )
        min_x, min_y = 10**9, 10**9
        max_x, max_y = -1, -1
        y_m = y_start
        for line in lines:
            if y_m + line_h > bottom_limit:
                break
            segments = _parse_subtitle_markup(line)
            has_markup = any(s[1] != "normal" for s in segments)
            if has_markup and fonts_styled.get("normal"):
                tw, th = _measure_line_segments(segments, fonts_styled, draw)
                px = max(0, min(w - tw, (w - tw) // 2))
                py = y_m
            else:
                try:
                    if hasattr(draw, "textbbox"):
                        bb = draw.textbbox((0, 0), line, font=font)
                    else:
                        bb = font.getbbox(line)
                    tw = bb[2] - bb[0]
                    th = bb[3] - bb[1]
                except Exception:
                    tw, th = 200, 24
                px = max(0, min(w - tw, (w - tw) // 2))
                py = y_m
            min_x = min(min_x, px - om)
            min_y = min(min_y, py - om)
            max_x = max(max_x, px + tw + om)
            max_y = max(max_y, py + th + om)
            y_m += line_h
        if max_x >= min_x and max_y >= min_y:
            x1b = max(0, min_x - pad)
            y1b = max(0, min_y - pad)
            x2b = min(w - 1, max_x + pad)
            y2b = min(h - 1, max_y + pad)
            draw.rectangle([x1b, y1b, x2b, y2b], fill=fill_rgb)
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


def _scale_blk_list_to_full_res(blk_list: List[TextBlock], scale: float) -> None:
    """Scale block coordinates (xyxy, lines) from downscaled frame back to full resolution. Modifies in place."""
    if not blk_list or scale <= 0 or abs(scale - 1.0) < 1e-6:
        return
    inv = 1.0 / scale
    for blk in blk_list:
        xyxy = getattr(blk, "xyxy", None)
        if xyxy is not None and len(xyxy) >= 4:
            blk.xyxy = [float(xyxy[i]) * inv if i < 4 else xyxy[i] for i in range(len(xyxy))]
        lines = getattr(blk, "lines", None)
        if lines:
            for line in lines:
                for pt in line:
                    if pt is not None and len(pt) >= 2:
                        pt[0] = float(pt[0]) * inv
                        pt[1] = float(pt[1]) * inv


def translate_video_textblk_list(
    translator: Any,
    blk_list: List[TextBlock],
    cfg: Any,
    frame_index: Optional[int] = None,
    img: Optional[np.ndarray] = None,
    flow_fixer: Any = None,
    use_flow_fixer_for_corrections: bool = False,
) -> None:
    """
    Video-only LLM stage: optional correct OCR, then translate blk_list in-place.

    Extracted from `run_one_frame_pipeline()` so the UI can overlap translation
    with detect/OCR/inpaint for later frames.
    """
    if translator is None or not blk_list:
        return

    # Strict pre-translate filter: skip OCR placeholders/noise entirely so they do not
    # consume LLM tokens or pollute cache/context. Keep stable empty translation for skipped lines.
    src_all = [blk.get_text() or "" for blk in blk_list]
    locked_idx = [i for i, s in enumerate(src_all) if _is_locked_watermark_line(s, cfg)]
    active_idx = [
        i for i, s in enumerate(src_all)
        if (i not in locked_idx) and (not _is_garbage_source(s))
    ]
    if not active_idx:
        for i, blk in enumerate(blk_list):
            if i in locked_idx:
                blk.translation = (blk.get_text() or "").strip()
            else:
                blk.translation = ""
        return
    if len(active_idx) != len(blk_list):
        for i, blk in enumerate(blk_list):
            if i in locked_idx:
                blk.translation = (blk.get_text() or "").strip()
            elif i not in active_idx:
                blk.translation = ""
    active_blks = [blk_list[i] for i in active_idx]

    # Optional LLM OCR correction (VideoCaptioner-style) before translation (only if translator has this param, e.g. LLM_API)
    correct_ocr = False
    try:
        params = getattr(translator, "params", None)
        if params is not None and "correct_ocr_with_llm" in params:
            try:
                correct_ocr = bool(translator.get_param_value("correct_ocr_with_llm"))
            except Exception:
                correct_ocr = False
    except Exception:
        correct_ocr = False

    if correct_ocr:
        try:
            texts = [blk.get_text() or "" for blk in active_blks]
            if any(t.strip() for t in texts):
                if use_flow_fixer_for_corrections and flow_fixer and getattr(flow_fixer, "request_completion", None):
                    glossary = (getattr(translator, "_video_glossary_hint", None) or "").strip() if translator else ""
                    lang_hint = None
                    if translator and getattr(translator, "lang_map", None) and getattr(translator, "lang_source", None):
                        lang_hint = (translator.lang_map.get(translator.lang_source, translator.lang_source) or "").strip()
                    corrected = correct_ocr_via_fixer(flow_fixer, texts, lang_hint=lang_hint, glossary=glossary)
                else:
                    corrected = translator.correct_ocr_texts(texts) if translator else texts
                if corrected and len(corrected) == len(active_blks):
                    for blk, new_text in zip(active_blks, corrected):
                        blk.text = [new_text] if isinstance(getattr(blk, "text", None), list) else new_text
                        if not getattr(blk, "lines", None):
                            blk.lines = []
        except Exception:
            pass

    # Temporal OCR stabilization: smooth per-frame OCR jitter before translation.
    _apply_temporal_ocr_stabilization(translator, active_blks, cfg, frame_index)

    configure_translator_video_nlp_parallel(translator, cfg)
    try:
        setattr(translator, "_current_page_key", "video_frame")
        setattr(translator, "_current_page_image", img)
        setattr(translator, "_video_current_sources", [blk.get_text() or "" for blk in active_blks])

        # Per-run cache keyed by (frame_segment, texts) so we never reuse a translation
        # from a later part of the video at the start (avoids "5-min subtitles at 0:00" bug).
        texts = [blk.get_text() or "" for blk in active_blks]
        segment_size = 300  # ~10s at 30fps; reuse only within same segment
        frame_segment = (frame_index // segment_size) if frame_index is not None else 0
        key = (frame_segment, tuple(texts))

        cache = getattr(translator, "_video_frame_cache", None)
        if cache is None:
            cache = {}
            setattr(translator, "_video_frame_cache", cache)

        cached = cache.get(key)

        # Recent-run cache: same source text in close succession (e.g. across segment boundary) reuses translation.
        if not (cached and len(cached) == len(active_blks)):
            recent = getattr(translator, "_video_recent_text_cache", None)
            if recent is None:
                recent = []
                setattr(translator, "_video_recent_text_cache", recent)

            texts_tup = tuple(texts)
            # Reuse only when source texts are non-garbage; otherwise placeholders like
            # "Line A"/"Text" can contaminate later frames with wrong cached translations.
            can_reuse_recent = bool(texts_tup) and all(not _is_garbage_source(t) for t in texts_tup)
            for (prev_texts, prev_trans) in recent:
                if not can_reuse_recent:
                    break
                if prev_texts == texts_tup and len(prev_trans) == len(active_blks):
                    cached = prev_trans
                    for blk, trans in zip(active_blks, cached):
                        blk.translation = trans
                    cache[key] = list(cached)
                    break
            else:
                cached = None

        if cached and len(cached) == len(active_blks):
            for blk, trans in zip(active_blks, cached):
                blk.translation = trans
            return

        prev_cached = cached if (cached and isinstance(cached, list)) else None
        do_reflection_via_fixer = (
            use_flow_fixer_for_corrections
            and flow_fixer
            and getattr(flow_fixer, "request_completion", None)
            and getattr(translator, "params", None)
            and "reflection_translation" in translator.params
            and translator.get_param_value("reflection_translation")
        )

        if do_reflection_via_fixer:
            translator.set_param_value("reflection_translation", False)

        translator.translate_textblk_lst(active_blks)

        # Guard against LLM glitches: if the model returns a count mismatch or blanks for some
        # items, do NOT overwrite a previously good cached translation with empty strings.
        new_list = []
        for i, blk in enumerate(active_blks):
            src_i = (texts[i] if i < len(texts) else (blk.get_text() or ""))
            t = _sanitize_subtitle_translation(
                src_i,
                str(getattr(blk, "translation", "") or "").strip(),
            )
            if (not t) and prev_cached and i < len(prev_cached):
                prev_t = _sanitize_subtitle_translation(src_i, str(prev_cached[i] or "").strip())
                if prev_t:
                    t = prev_t
                    blk.translation = t
            if not t:
                # Last resort: keep something stable so UI/compositing doesn't flicker off.
                src_fallback = (blk.get_text() or "").strip()
                t = "" if _is_garbage_source(src_fallback) else src_fallback
                blk.translation = t
            new_list.append(t)

        if do_reflection_via_fixer and new_list:
            to_lang = (getattr(translator, "lang_map", None) or {}).get(
                getattr(translator, "lang_target", None),
                getattr(translator, "lang_target", "en"),
            ) or "en"
            if not isinstance(to_lang, str):
                to_lang = "en"
            to_lang = str(to_lang).strip() or "en"
            glossary = (getattr(translator, "_video_glossary_hint", None) or "").strip()
            improved = reflect_translations_via_fixer(flow_fixer, texts, new_list, to_lang=to_lang, glossary=glossary)
            if improved and len(improved) == len(active_blks):
                new_list = list(improved)
                for blk, t in zip(active_blks, new_list):
                    blk.translation = t
            translator.set_param_value("reflection_translation", True)

        # If we have mixed garbage source + real text and likely swap, reassign by length.
        if len(active_blks) >= 2 and len(new_list) == len(active_blks):
            src_lens = [len((blk.get_text() or "").strip()) for blk in active_blks]
            has_garbage = any(_is_garbage_source(blk.get_text() or "") for blk in active_blks)
            trans_lens = [len(t) for t in new_list]
            # Only reassign when correlation looks inverted: shortest source has longer trans than longest source.
            order_src = sorted(range(len(active_blks)), key=lambda i: src_lens[i])
            order_trans = sorted(range(len(new_list)), key=lambda i: trans_lens[i])
            min_src_idx = order_src[0]
            max_src_idx = order_src[-1]
            min_trans_idx = order_trans[0]
            max_trans_idx = order_trans[-1]
            likely_swap = (
                has_garbage
                and trans_lens[min_src_idx] > trans_lens[max_src_idx]
                and src_lens[min_src_idx] < src_lens[max_src_idx]
            )
            if likely_swap:
                reordered = [None] * len(new_list)
                for (i_src, i_trans) in zip(order_src, order_trans):
                    reordered[i_src] = new_list[i_trans]
                new_list = reordered
                for blk, t in zip(active_blks, new_list):
                    blk.translation = t
                LOGGER.debug(
                    "Video translator: reassigned %d block translations by length (garbage source detected).",
                    len(active_blks),
                )

        cache[key] = new_list

        # Feed recent-run cache so same text in close succession reuses (avoid duplicate API calls).
        recent = getattr(translator, "_video_recent_text_cache", None)
        if recent is not None and texts and all(not _is_garbage_source(t) for t in texts):
            recent.append((tuple(texts), list(new_list)))
            while len(recent) > 30:
                recent.pop(0)

        try:
            msg = "; ".join(
                (blk.get_text() or "").strip() + " -> " + str(blk.translation or "").replace("\n", " ").strip()
                for blk in active_blks
            )
        except Exception:
            msg = ""

        if msg:
            LOGGER.info("Video translator: translated %d blocks: %s", len(active_blks), msg)
        else:
            LOGGER.info("Video translator: translated %d blocks.", len(active_blks))

    except Exception:
        # Last resort: keep stable values so UI/compositing doesn't flicker.
        for blk in active_blks:
            if getattr(blk, "translation", None) is None or str(blk.translation).strip() == "":
                blk.translation = blk.get_text() or ""
    finally:
        try:
            delattr(translator, "_video_current_sources")
        except Exception:
            pass
        clear_translator_video_nlp_parallel(translator)


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
    detect_roi_xyxy: Optional[List[int]] = None,
    draw_subtitles: bool = True,
    frame_index: Optional[int] = None,
    flow_fixer: Any = None,
    use_flow_fixer_for_corrections: bool = False,
) -> Tuple[np.ndarray, List[TextBlock], Optional[np.ndarray]]:
    """
    Run detect -> OCR -> translate -> inpaint on a single frame. Returns (out_img, blk_list, mask).
    If skip_detect and region preset is not full, uses a single fixed block for the subtitle band (no detector).
    When draw_subtitles is False (inpaint-only soft subs), inpainted frame is returned without drawing text.
    If detect_roi_xyxy is provided, the detector runs on that cropped ROI and resulting blocks are offset back to full-frame coordinates.
    Video inpaint runs only when enable_ocr is True, an OCR module is loaded, and at least one block has substantive OCR after filtering (no inpaint on empty-text frames).
    """
    cfg = cfg or pcfg.module
    _t0 = time.perf_counter()
    _t_detect = _t0
    _t_ocr = _t0
    _t_translate = _t0
    _t_inpaint = _t0
    _t_draw = _t0
    im_h, im_w = img.shape[:2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    blk_list: List[TextBlock] = []
    mask: Optional[np.ndarray] = None
    frac = _region_fraction_from_preset(cfg)
    native_bottom_band = bool(
        skip_detect
        and frac > 0
        and bool(getattr(cfg, "video_translator_bottom_band_native_mode", False))
    )
    subtitle_black_box_mode = bool(getattr(cfg, "video_translator_subtitle_black_box_mode", False))

    if skip_detect and frac > 0:
        blk_list = [_make_fixed_region_block(im_w, im_h, frac)]
    elif enable_detect and detector is not None:
        try:
            if detect_roi_xyxy and len(detect_roi_xyxy) >= 4:
                x1, y1, x2, y2 = [int(v) for v in detect_roi_xyxy[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(im_w, x2), min(im_h, y2)
                if x2 > x1 and y2 > y1:
                    roi_img = img[y1:y2, x1:x2]
                    roi_h, roi_w = roi_img.shape[:2]
                    if max(roi_w, roi_h) > _DETECT_MAX_SIDE:
                        scale_roi = _DETECT_MAX_SIDE / max(roi_w, roi_h)
                        sw, sh = int(roi_w * scale_roi), int(roi_h * scale_roi)
                        if sw > 0 and sh > 0:
                            small_roi = cv2.resize(roi_img, (sw, sh), interpolation=cv2.INTER_LINEAR)
                            mask_roi, blk_list = detector.detect(small_roi, None)
                            _scale_blk_list_to_full_res(blk_list or [], scale_roi)
                        else:
                            mask_roi, blk_list = detector.detect(roi_img, None)
                    else:
                        mask_roi, blk_list = detector.detect(roi_img, None)
                    if not blk_list:
                        # ROI detector can miss; fall back to full-frame detect to avoid quality drops.
                        mask, blk_list = detector.detect(img, None)
                    else:
                        mask = None
                        if mask_roi is not None:
                            try:
                                # Best-effort: expand ROI mask back to full frame for debugging/return value.
                                if isinstance(mask_roi, np.ndarray) and mask_roi.ndim >= 2:
                                    mask_full = np.zeros((im_h, im_w), dtype=mask_roi.dtype)
                                    mask_full[y1:y2, x1:x2] = (
                                        mask_roi
                                        if mask_roi.shape[0] == (y2 - y1)
                                        else cv2.resize(mask_roi, (x2 - x1, y2 - y1))
                                    )
                                    mask = mask_full
                            except Exception:
                                mask = None
                        # Offset detector outputs back to full-frame coordinates.
                        for blk in (blk_list or []):
                            xyxy = getattr(blk, "xyxy", None)
                            if xyxy is not None and len(xyxy) >= 4:
                                base_xyxy = [xyxy[0] + x1, xyxy[1] + y1, xyxy[2] + x1, xyxy[3] + y1]
                                blk.xyxy = base_xyxy + list(xyxy[4:]) if len(xyxy) > 4 else base_xyxy
                            lines = getattr(blk, "lines", None)
                            if lines:
                                try:
                                    for line in lines:
                                        for pt in line:
                                            if pt is not None and len(pt) >= 2:
                                                pt[0] += x1
                                                pt[1] += y1
                                except Exception:
                                    pass
                else:
                    mask, blk_list = detector.detect(img, None)
            else:
                # Run detector on downscaled frame when large (faster; OCR still uses full-res crops).
                max_side = max(im_w, im_h)
                if max_side > _DETECT_MAX_SIDE:
                    scale = _DETECT_MAX_SIDE / max_side
                    small_w, small_h = int(im_w * scale), int(im_h * scale)
                    if small_w > 0 and small_h > 0:
                        small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                        mask, blk_list = detector.detect(small_img, None)
                        _scale_blk_list_to_full_res(blk_list or [], scale)
                        mask = None  # Inpaint mask is built from blocks, not detector mask
                    else:
                        mask, blk_list = detector.detect(img, None)
                else:
                    mask, blk_list = detector.detect(img, None)
            if blk_list:
                for blk in blk_list:
                    if getattr(blk, "lines", None) and len(blk.lines) > 0:
                        examine_textblk(blk, im_w, im_h, sort=True)
                blk_list = remove_contained_boxes(blk_list)
                blk_list = deduplicate_primary_boxes(blk_list, iou_threshold=0.5)
                _sanitize_blocks_for_ocr(blk_list, im_w, im_h)
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
    _t_detect = time.perf_counter()

    if not blk_list:
        return img.copy(), [], None

    # When skip_detect uses a single full-band block, inpainting the entire band produces a solid color box.
    # Skip inpainting and instead draw a semi-opaque bar + text so the region looks like a subtitle bar.
    skip_inpaint_for_fixed_band = bool(
        skip_detect and frac > 0 and len(blk_list) == 1
        and getattr(blk_list[0], "xyxy", None) and len(blk_list[0].xyxy) >= 4
        and int(blk_list[0].xyxy[0]) == 0 and int(blk_list[0].xyxy[2]) == im_w
        and int(blk_list[0].xyxy[3]) == im_h
    ) and (not native_bottom_band)

    # Inpaint is decided after OCR (see below): geometry-only boxes with no real text must not
    # trigger inpainting (especially skip_detect bottom-band = whole strip erased).
    # Optional overlap: run inpainting in a background thread while OCR/translation executes.
    do_inpaint = False
    inpaint_future = None
    inpaint_executor = None
    mask_out: Optional[np.ndarray] = None

    def _do_inpaint(img_in: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run inpainting step only; returns (inpainted_img, mask_used)."""
        img_local = img_in
        mask_local: Optional[np.ndarray] = None
        if not do_inpaint:
            return img_local, mask_local

        try:
            # Video inpaint caching: only run inpaint when subtitle content/region changes.
            try:
                keys = []
                for b in blk_list:
                    k = _compute_ocr_cache_key(b, img_local)
                    if k is not None:
                        keys.append(k)
                if keys:
                    # v3: per-block neural inpaint (blk_list passed); not bottom-band full-image crop.
                    inpaint_key = ("v3", tuple(keys))
                else:
                    texts = tuple((b.get_text() or "").strip() for b in blk_list)
                    inpaint_key = ("v3", "text", texts)
            except Exception:
                inpaint_key = None

            if inpaint_key is not None:
                last_key = getattr(inpainter, "_video_frame_inpaint_cache_key", None)
                last_img = getattr(inpainter, "_video_frame_inpaint_cache_img", None)
                last_mask = getattr(inpainter, "_video_frame_inpaint_cache_mask", None)
                if last_key == inpaint_key and last_img is not None:
                    out = img_local.copy()
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
                            out = img_local.copy()
                    except Exception:
                        out = img_local.copy()
                    return out, last_mask

            # Build inpaint mask. In skip-detect mode, the subtitle band block is deterministic,
            # so we can cache the dilated mask for each (w,h,frac) to reduce per-frame CPU work.
            # Slightly stronger base dilation than comic pipeline — video subs have halos/outlines.
            dilate_px = max(3, min(12, min(im_w, im_h) // 150))
            mask_already_dilated = False
            fixed_mask = None
            mask_local = None

            if skip_detect and frac > 0 and len(blk_list) == 1:
                try:
                    expected_y_min = int(im_h * (1.0 - frac))
                    expected_y_min = max(0, min(expected_y_min, im_h - 1))
                    xyxy0 = getattr(blk_list[0], "xyxy", None)
                    if (
                        xyxy0
                        and len(xyxy0) >= 4
                        and int(xyxy0[0]) == 0
                        and int(xyxy0[1]) == expected_y_min
                        and int(xyxy0[2]) == im_w
                        and int(xyxy0[3]) == im_h
                    ):
                        cache_key = ("v1", im_w, im_h, float(frac), int(dilate_px))
                        fixed_mask = _FIXED_REGION_INPAINT_MASK_CACHE.get(cache_key)
                        if fixed_mask is None:
                            mask_base = build_mask_with_resolved_overlaps(
                                blk_list, im_w, im_h, text_blocks_for_nudge=None
                            )
                            if mask_base is not None and mask_base.size > 0:
                                kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
                                fixed_mask = cv2.dilate((mask_base > 127).astype(np.uint8), kernel)
                                fixed_mask = (fixed_mask > 0).astype(np.uint8) * 255
                                _FIXED_REGION_INPAINT_MASK_CACHE[cache_key] = fixed_mask
                                _FIXED_REGION_INPAINT_MASK_CACHE_ORDER.append(cache_key)
                                if (
                                    len(_FIXED_REGION_INPAINT_MASK_CACHE_ORDER)
                                    > _FIXED_REGION_INPAINT_MASK_CACHE_MAX
                                ):
                                    old_key = _FIXED_REGION_INPAINT_MASK_CACHE_ORDER.pop(0)
                                    _FIXED_REGION_INPAINT_MASK_CACHE.pop(old_key, None)
                        if fixed_mask is not None:
                            mask_local = fixed_mask
                            mask_already_dilated = True
                except Exception:
                    fixed_mask = None

            if mask_local is None:
                from modules.textdetector.outside_text_processor import OSB_LABELS
                text_for_nudge = [
                    b
                    for b in blk_list
                    if (getattr(b, "label", None) or "").strip().lower() in OSB_LABELS
                ]
                mask_local = build_mask_with_resolved_overlaps(
                    blk_list,
                    im_w,
                    im_h,
                    text_blocks_for_nudge=text_for_nudge if text_for_nudge else None,
                )
                # Some object detectors can provide malformed polygon lines that explode mask area.
                # If resolved-overlap mask is suspiciously larger than the rectangle footprint, fallback.
                try:
                    if mask_local is not None and mask_local.size > 0 and blk_list:
                        frame_area = float(im_w * im_h)
                        mask_area = float(np.count_nonzero(mask_local > 127))
                        rect_area = 0.0
                        for b in blk_list:
                            xyxy = getattr(b, "xyxy", None)
                            if not xyxy or len(xyxy) < 4:
                                continue
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            x1, x2 = max(0, min(x1, im_w)), max(0, min(x2, im_w))
                            y1, y2 = max(0, min(y1, im_h)), max(0, min(y2, im_h))
                            if x2 > x1 and y2 > y1:
                                rect_area += float((x2 - x1) * (y2 - y1))
                        too_large_vs_rects = rect_area > 0 and mask_area > (rect_area * 3.0)
                        too_large_vs_frame = frame_area > 0 and (mask_area / frame_area) > 0.35
                        if too_large_vs_rects and too_large_vs_frame:
                            LOGGER.debug(
                                "Video inpaint mask fallback: suspicious mask area %.0f vs rect area %.0f (blocks=%d).",
                                mask_area,
                                rect_area,
                                int(len(blk_list)),
                            )
                            mask_local = _fallback_xyxy_mask(blk_list, im_w, im_h)
                except Exception:
                    pass

            # Native bottom-band mode: in skip-detect subtitle band workflows, force a full
            # rectangular mask for the configured bottom fraction (instead of line polygons).
            if native_bottom_band and skip_detect and frac > 0 and len(blk_list) == 1:
                try:
                    y_min = int(im_h * (1.0 - frac))
                    y_min = max(0, min(y_min, im_h - 1))
                    mask_local = np.zeros((im_h, im_w), dtype=np.uint8)
                    mask_local[y_min:im_h, :] = 255
                except Exception:
                    pass

            if mask_local is not None and mask_local.size > 0:
                if not mask_already_dilated:
                    kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
                    mask_local = cv2.dilate((mask_local > 127).astype(np.uint8), kernel)
                    mask_local = (mask_local > 0).astype(np.uint8) * 255

                if bool(getattr(cfg, "video_translator_inpaint_subtitle_mask_expand", True)):
                    mask_local = _video_subtitle_inpaint_mask_expand(
                        mask_local, im_h, im_w, dilate_px
                    )

                # Pass detected blocks so InpainterBase runs neural fill per box (enlarged_window crop
                # around each xyxy only). inpaint_full_image=True uses the whole frame + mask instead.
                blk_list_arg = None if getattr(cfg, "inpaint_full_image", False) else blk_list
                # OpenCV video frames are BGR; LaMa / most neural inpainters are trained on RGB.
                if bool(getattr(cfg, "video_translator_inpaint_bgr_to_rgb", True)):
                    img_rgb = cv2.cvtColor(img_local, cv2.COLOR_BGR2RGB)
                    out_rgb = inpainter.inpaint(img_rgb, mask_local, blk_list_arg)
                    if out_rgb is not None and out_rgb.shape == img_local.shape:
                        img_local = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                    else:
                        img_local = inpainter.inpaint(img_local, mask_local, blk_list_arg)
                else:
                    img_local = inpainter.inpaint(img_local, mask_local, blk_list_arg)

                if inpaint_key is not None and img_local is not None:
                    try:
                        setattr(inpainter, "_video_frame_inpaint_cache_key", inpaint_key)
                        setattr(inpainter, "_video_frame_inpaint_cache_img", img_local.copy() if hasattr(img_local, "copy") else img_local)
                        setattr(inpainter, "_video_frame_inpaint_cache_mask", mask_local.copy() if hasattr(mask_local, "copy") else mask_local)
                    except Exception:
                        pass
        except Exception:
            pass

        return img_local, mask_local

    if enable_ocr and ocr is not None:
        try:
            from modules.ocr.base import ensure_blocks_have_lines, normalize_block_text
            ensure_blocks_have_lines(blk_list)
            # skip_detect optimization: in fixed bottom-band mode, run OCR only when the band changed enough.
            # This avoids per-frame OCR on static subtitles / mild codec noise.
            skip_ocr_by_band_diff = False
            if skip_detect and frac > 0 and len(blk_list) == 1:
                try:
                    diff_thr = float(
                        getattr(cfg, "video_translator_skip_detect_ocr_diff_threshold", 2.0) or 2.0
                    )
                    diff_thr = max(0.25, min(24.0, diff_thr))
                except (TypeError, ValueError):
                    diff_thr = 2.0
                prev_thumb = getattr(ocr, "_video_skip_detect_prev_band_thumb", None)
                prev_text = str(getattr(ocr, "_video_skip_detect_prev_band_text", "") or "").strip()
                cur_thumb = _compute_skip_detect_band_thumb(img, frac)
                if (
                    isinstance(prev_thumb, np.ndarray)
                    and isinstance(cur_thumb, np.ndarray)
                    and prev_thumb.shape == cur_thumb.shape
                    and prev_text
                ):
                    try:
                        band_diff = float(
                            np.mean(
                                np.abs(
                                    cur_thumb.astype(np.int16) - prev_thumb.astype(np.int16)
                                )
                            )
                        )
                    except Exception:
                        band_diff = diff_thr + 1.0
                    if band_diff < diff_thr:
                        _set_block_text(blk_list[0], prev_text)
                        skip_ocr_by_band_diff = True
                if isinstance(cur_thumb, np.ndarray):
                    setattr(ocr, "_video_skip_detect_prev_band_thumb", cur_thumb)
                else:
                    setattr(ocr, "_video_skip_detect_prev_band_thumb", None)

            if skip_ocr_by_band_diff:
                _t_ocr = time.perf_counter()
            else:
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
                pending_geo_keys: List[Optional[Tuple[int, int, int, int]]] = []
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
                        # General temporal OCR reuse (all modes): if same quantized region changed only
                        # a little since previous frame, reuse previous OCR text and skip expensive OCR.
                        reused_temporal = False
                        try:
                            xyxy = getattr(blk, "xyxy", None)
                            if xyxy and len(xyxy) >= 4:
                                q = max(2, min(32, int(getattr(cfg, "video_translator_ocr_cache_geo_quantization", 8) or 8)))
                                geo_key = (
                                    int(round(float(xyxy[0]) / q)),
                                    int(round(float(xyxy[1]) / q)),
                                    int(round(float(xyxy[2]) / q)),
                                    int(round(float(xyxy[3]) / q)),
                                )
                                temporal = getattr(ocr, "_video_temporal_ocr_box_cache", None)
                                if temporal is None:
                                    temporal = {}
                                    setattr(ocr, "_video_temporal_ocr_box_cache", temporal)
                                entry = temporal.get(geo_key)
                                if entry and isinstance(entry, tuple) and len(entry) == 2:
                                    prev_thumb, prev_text = entry
                                    cur_thumb = _compute_block_thumb(img, blk)
                                    if (
                                        isinstance(prev_thumb, np.ndarray)
                                        and isinstance(cur_thumb, np.ndarray)
                                        and prev_thumb.shape == cur_thumb.shape
                                        and str(prev_text or "").strip()
                                    ):
                                        diff_thr = float(
                                            getattr(cfg, "video_translator_ocr_temporal_diff_threshold", 2.0) or 2.0
                                        )
                                        diff_thr = max(0.25, min(24.0, diff_thr))
                                        diff = float(
                                            np.mean(
                                                np.abs(
                                                    cur_thumb.astype(np.int16) - prev_thumb.astype(np.int16)
                                                )
                                            )
                                        )
                                        if diff < diff_thr:
                                            _set_block_text(blk, str(prev_text or "").strip())
                                            n_hits += 1
                                            reused_temporal = True
                                if not reused_temporal:
                                    need_ocr.append(blk)
                                    pending_keys.append(key)
                                    pending_geo_keys.append(geo_key)
                            else:
                                need_ocr.append(blk)
                                pending_keys.append(key)
                                pending_geo_keys.append(None)
                        except Exception:
                            need_ocr.append(blk)
                            pending_keys.append(key)
                            pending_geo_keys.append(None)
                if need_ocr:
                    ocr.run_ocr(img, need_ocr)
                    for blk, key, geo_key in zip(need_ocr, pending_keys, pending_geo_keys):
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
                                    ev_total = int(getattr(ocr, "_video_ocr_cache_evictions_since_log", 0) or 0) + evicted
                                    setattr(ocr, "_video_ocr_cache_evictions_since_log", ev_total)
                                    now = time.monotonic()
                                    last = float(getattr(ocr, "_video_ocr_cache_evictions_last_log_ts", 0.0) or 0.0)
                                    # Periodic summary instead of one log per eviction event.
                                    if ev_total >= 100 or (last > 0.0 and (now - last) >= 8.0):
                                        LOGGER.debug(
                                            "Video OCR cache: evicted %d entries total in this period (max %d, cache size %d)",
                                            ev_total, MAX_VIDEO_OCR_CACHE_SIZE, len(cache),
                                        )
                                        setattr(ocr, "_video_ocr_cache_evictions_since_log", 0)
                                        setattr(ocr, "_video_ocr_cache_evictions_last_log_ts", now)
                                    elif last <= 0.0:
                                        setattr(ocr, "_video_ocr_cache_evictions_last_log_ts", now)
                                if key not in cache:
                                    cache_order.append(key)
                                cache[key] = txt
                            # Update temporal per-box OCR reuse cache after a real OCR run.
                            if geo_key is not None and txt.strip():
                                temporal = getattr(ocr, "_video_temporal_ocr_box_cache", None)
                                if temporal is None:
                                    temporal = {}
                                    setattr(ocr, "_video_temporal_ocr_box_cache", temporal)
                                thumb = _compute_block_thumb(img, blk)
                                if isinstance(thumb, np.ndarray):
                                    temporal[geo_key] = (thumb, txt.strip())
                                    if len(temporal) > 1024:
                                        try:
                                            temporal.pop(next(iter(temporal)))
                                        except Exception:
                                            pass
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
                if skip_detect and frac > 0 and len(blk_list) == 1:
                    try:
                        current_text = (blk_list[0].get_text() or "").strip()
                        if current_text:
                            setattr(ocr, "_video_skip_detect_prev_band_text", current_text)
                    except Exception:
                        pass
        except Exception:
            pass
    _t_ocr = time.perf_counter()

    # Drop blocks with no substantive OCR so we do not inpaint empty bands or noise boxes.
    if enable_ocr and ocr is not None and blk_list:
        n_det_blocks = len(blk_list)
        blk_list = [
            b
            for b in blk_list
            if _is_substantive_ocr_for_video_inpaint(_block_plain_ocr_text(b), n_det_blocks)
        ]

    # Inpaint only when OCR is on and at least one block has substantive text (blk_list was
    # filtered above). Never inpaint on detector-only / empty-OCR frames.
    do_inpaint = bool(
        enable_inpaint
        and inpainter is not None
        and enable_ocr
        and ocr is not None
        and blk_list
        and not skip_inpaint_for_fixed_band
        and not subtitle_black_box_mode
    )

    if do_inpaint:
        overlap = bool(getattr(cfg, "video_translator_overlap_inpaint", False))
        require_cpu = bool(getattr(cfg, "video_translator_overlap_inpaint_require_cpu", True))
        if overlap:
            if require_cpu:
                dev = str(getattr(inpainter, "device", "") or "").strip().lower()
                if dev and dev != "cpu":
                    overlap = False
                if not dev:
                    overlap = False
            if overlap:
                import concurrent.futures
                inpaint_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                # Use a copy to avoid any potential mutation issues across threads.
                inpaint_future = inpaint_executor.submit(_do_inpaint, img.copy() if hasattr(img, "copy") else img)

    # NOTE: LLM OCR correction (and translation) are handled inside
    # `translate_video_textblk_list()` when `enable_translate=True`.

    if enable_translate and translator is not None:
        translate_video_textblk_list(
            translator=translator,
            blk_list=blk_list,
            cfg=cfg,
            frame_index=frame_index,
            img=img,
            flow_fixer=flow_fixer,
            use_flow_fixer_for_corrections=use_flow_fixer_for_corrections,
        )
    _t_translate = time.perf_counter()

    # Wait for background inpainting (if started), otherwise run it synchronously here.
    if do_inpaint:
        try:
            if inpaint_future is not None:
                try:
                    img, mask = inpaint_future.result()
                except Exception:
                    img, mask = _do_inpaint(img)
            else:
                img, mask = _do_inpaint(img)
        finally:
            if inpaint_executor is not None:
                try:
                    inpaint_executor.shutdown(wait=False)
                except Exception:
                    pass
    _t_inpaint = time.perf_counter()

    out = img.copy() if img is not None else np.zeros((im_h, im_w, 3), dtype=np.uint8)
    if draw_subtitles and blk_list:
        # Flicker guard (one-pass burn-in): if current frame has no drawable subtitle due to
        # transient OCR/translation miss, briefly reuse previous frame subtitle state.
        # Keep this window short to avoid stale subtitles lingering.
        hold_frames = 2
        has_drawable = _has_drawable_subtitles(blk_list)
        if not has_drawable and frame_index is not None and translator is not None:
            prev = getattr(translator, "_video_last_drawn_blk_state", None)
            if prev and isinstance(prev, tuple) and len(prev) == 2:
                prev_idx, prev_state = prev
                if isinstance(prev_idx, int) and isinstance(prev_state, list):
                    if 0 < (frame_index - prev_idx) <= hold_frames:
                        try:
                            reused = []
                            for b in prev_state:
                                nb = TextBlock()
                                nb.xyxy = list(getattr(b, "xyxy", None) or [])
                                nb.lines = getattr(b, "lines", None)
                                nb.text = getattr(b, "text", None)
                                nb.translation = getattr(b, "translation", "")
                                reused.append(nb)
                            if reused:
                                blk_list = reused
                                has_drawable = True
                        except Exception:
                            pass

        # When we skipped inpainting for the fixed band (skip_detect), draw a semi-opaque bar so original text is covered.
        # Per-block black-box mode covers only detected/wrapped subtitle areas instead of the full band width.
        if (
            skip_inpaint_for_fixed_band
            and blk_list[0].xyxy
            and len(blk_list[0].xyxy) >= 4
            and not subtitle_black_box_mode
        ):
            y1 = int(blk_list[0].xyxy[1])
            y2 = int(blk_list[0].xyxy[3])
            y1, y2 = max(0, y1), min(im_h, y2)
            if y2 > y1:
                overlay = out.copy()
                cv2.rectangle(overlay, (0, y1), (im_w, y2), (0, 0, 0), -1)
                alpha = 0.65
                cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
        style = (getattr(cfg, "video_translator_subtitle_style", None) or "default").strip().lower()
        if style not in ("anime", "documentary"):
            style = "default"
        try:
            bb_pad = int(getattr(cfg, "video_translator_subtitle_black_box_padding", 6) or 0)
        except (TypeError, ValueError):
            bb_pad = 6
        bb_pad = max(0, min(64, bb_pad))
        bb_bgr = (
            int(getattr(cfg, "video_translator_subtitle_black_box_b", 0)),
            int(getattr(cfg, "video_translator_subtitle_black_box_g", 0)),
            int(getattr(cfg, "video_translator_subtitle_black_box_r", 0)),
        )
        _draw_text_on_image(
            out,
            blk_list,
            style=style,
            black_box_behind_text=subtitle_black_box_mode,
            black_box_padding=bb_pad,
            black_box_bgr=bb_bgr,
        )
        if frame_index is not None and translator is not None and _has_drawable_subtitles(blk_list):
            try:
                snapshot = []
                for b in blk_list:
                    raw_src = b.get_text() if callable(getattr(b, "get_text", None)) else getattr(b, "text", "")
                    src = (raw_src if isinstance(raw_src, str) else " ".join(str(x or "") for x in (raw_src or []))).strip()
                    trans = (getattr(b, "translation", None) or "").strip()
                    if _is_garbage_ocr_source(src) or not trans:
                        continue
                    nb = TextBlock()
                    nb.xyxy = list(getattr(b, "xyxy", None) or [])
                    nb.lines = getattr(b, "lines", None)
                    nb.text = getattr(b, "text", None)
                    nb.translation = trans
                    snapshot.append(nb)
                if snapshot:
                    setattr(translator, "_video_last_drawn_blk_state", (int(frame_index), snapshot))
            except Exception:
                pass
    _t_draw = time.perf_counter()
    try:
        # Keep debug concise: log slow frames, and periodic samples for visibility.
        total_ms = (_t_draw - _t0) * 1000.0
        should_log = (total_ms >= 600.0) or (frame_index is not None and int(frame_index) % 30 == 0)
        if should_log:
            LOGGER.debug(
                "Video frame timing idx=%s total=%.1fms detect=%.1fms ocr=%.1fms translate=%.1fms inpaint=%.1fms draw=%.1fms blocks=%d",
                str(frame_index) if frame_index is not None else "?",
                total_ms,
                (_t_detect - _t0) * 1000.0,
                (_t_ocr - _t_detect) * 1000.0,
                (_t_translate - _t_ocr) * 1000.0,
                (_t_inpaint - _t_translate) * 1000.0,
                (_t_draw - _t_inpaint) * 1000.0,
                int(len(blk_list or [])),
            )
        # Periodic identity log helps detect accidental model/object re-instantiation.
        id_log_every = int(getattr(cfg, "video_translator_debug_id_log_every", 60) or 60)
        if id_log_every > 0 and frame_index is not None and int(frame_index) % id_log_every == 0:
            det_model = getattr(detector, "model", None) if detector is not None else None
            ocr_model = getattr(ocr, "model", None) if ocr is not None else None
            ocr_v5_model = getattr(ocr, "v5_rec_model", None) if ocr is not None else None
            LOGGER.debug(
                "Video object ids idx=%s detector_id=%s detector_model_id=%s ocr_id=%s ocr_model_id=%s ocr_v5_model_id=%s",
                str(frame_index),
                hex(id(detector)) if detector is not None else "None",
                hex(id(det_model)) if det_model is not None else "None",
                hex(id(ocr)) if ocr is not None else "None",
                hex(id(ocr_model)) if ocr_model is not None else "None",
                hex(id(ocr_v5_model)) if ocr_v5_model is not None else "None",
            )
    except Exception:
        pass
    return out, blk_list, mask
