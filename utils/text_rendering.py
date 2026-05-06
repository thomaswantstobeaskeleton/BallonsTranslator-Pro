from __future__ import annotations

import math
import re
import sys
import unicodedata
import ctypes.util
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


WRITING_MODE_AUTO = "auto"
WRITING_MODE_HORIZONTAL_LTR = "horizontal_ltr"
WRITING_MODE_VERTICAL_RL = "vertical_rl"
WRITING_MODE_RTL = "rtl"
WRITING_MODES = {
    WRITING_MODE_AUTO,
    WRITING_MODE_HORIZONTAL_LTR,
    WRITING_MODE_VERTICAL_RL,
    WRITING_MODE_RTL,
}

FIT_MODE_SHRINK = "shrink"
FIT_MODE_EXPAND = "expand"
FIT_MODE_PRESERVE = "preserve"
FIT_MODE_BALANCE = "balance"
FIT_MODES = {FIT_MODE_SHRINK, FIT_MODE_EXPAND, FIT_MODE_PRESERVE, FIT_MODE_BALANCE}

LINE_BREAK_AUTO = "auto"
LINE_BREAK_CJK_STRICT = "cjk_strict"
LINE_BREAK_BALANCED = "balanced"
LINE_BREAK_LOOSE = "loose"
LINE_BREAK_STRATEGIES = {LINE_BREAK_AUTO, LINE_BREAK_CJK_STRICT, LINE_BREAK_BALANCED, LINE_BREAK_LOOSE}

CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
KOREAN_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
RTL_RE = re.compile(r"[\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
OPENING_PUNCT = set("([{（［｛〈《「『【〔〖〝‘“")
CLOSING_PUNCT = set(")]},.!?:;、。，．！？：；⁈⁉‼⁇…）］｝〉》」』】〕〗〟’”")
VERTICAL_PUNCT_MAP = {
    "?!": "⁈",
    "!?": "⁉",
    "!!": "‼",
    "??": "⁇",
    "...": "…",
    "……": "…",
    "，": "、",
}
VERTICAL_CENTER_PUNCT = set("、。，．・：；！？⁈⁉‼⁇…⋯")
VERTICAL_ROTATE_PUNCT = set("—―ー～-()[]{}<>（）［］｛｝〈〉《》「」『』【】〔〕")

MANGA_PRESETS = {
    "default_manga_bubble": {
        "label": "Default manga bubble",
        "writing_mode": WRITING_MODE_AUTO,
        "fit_mode": FIT_MODE_SHRINK,
        "font_size": 24.0,
        "stroke_width": 0.08,
        "line_spacing": 1.12,
        "letter_spacing": 1.05,
        "alignment": 1,
        "text_padding": 2.0,
        "line_break_strategy": LINE_BREAK_AUTO,
    },
    "vertical_cjk_bubble": {
        "label": "Vertical JP/CN bubble",
        "writing_mode": WRITING_MODE_VERTICAL_RL,
        "fit_mode": FIT_MODE_SHRINK,
        "font_size": 22.0,
        "stroke_width": 0.06,
        "line_spacing": 1.08,
        "letter_spacing": 1.0,
        "alignment": 1,
        "text_padding": 2.0,
        "line_break_strategy": LINE_BREAK_CJK_STRICT,
    },
    "sfx_bold": {
        "label": "SFX bold",
        "writing_mode": WRITING_MODE_AUTO,
        "fit_mode": FIT_MODE_EXPAND,
        "font_size": 36.0,
        "stroke_width": 0.14,
        "line_spacing": 1.0,
        "letter_spacing": 1.08,
        "alignment": 1,
        "bold": True,
        "text_padding": 3.0,
        "line_break_strategy": LINE_BREAK_LOOSE,
    },
    "caption_box": {
        "label": "Caption/narration box",
        "writing_mode": WRITING_MODE_HORIZONTAL_LTR,
        "fit_mode": FIT_MODE_BALANCE,
        "font_size": 20.0,
        "stroke_width": 0.0,
        "line_spacing": 1.18,
        "letter_spacing": 1.0,
        "alignment": 0,
        "text_padding": 4.0,
        "line_break_strategy": LINE_BREAK_BALANCED,
    },
    "small_aside": {
        "label": "Small aside text",
        "writing_mode": WRITING_MODE_AUTO,
        "fit_mode": FIT_MODE_PRESERVE,
        "font_size": 14.0,
        "stroke_width": 0.04,
        "line_spacing": 1.05,
        "letter_spacing": 1.0,
        "alignment": 1,
        "text_padding": 1.0,
        "line_break_strategy": LINE_BREAK_AUTO,
    },
}


@dataclass
class TextRenderDiagnostics:
    resolved_writing_mode: str
    overflow: bool = False
    measured_bounds: Tuple[float, float] = (0.0, 0.0)
    box_bounds: Tuple[float, float] = (0.0, 0.0)
    missing_glyphs: List[str] = None
    fallback_chain: str = ""
    fit_mode: str = FIT_MODE_SHRINK
    adjusted_font_size: float = 0.0
    line_count: int = 0
    column_count: int = 0
    line_break_strategy: str = LINE_BREAK_AUTO

    def to_dict(self) -> dict:
        return {
            "resolved_writing_mode": self.resolved_writing_mode,
            "overflow": bool(self.overflow),
            "measured_bounds": list(self.measured_bounds),
            "box_bounds": list(self.box_bounds),
            "missing_glyphs": list(self.missing_glyphs or []),
            "fallback_chain": self.fallback_chain,
            "fit_mode": self.fit_mode,
            "adjusted_font_size": self.adjusted_font_size,
            "line_count": self.line_count,
            "column_count": self.column_count,
            "line_break_strategy": self.line_break_strategy,
        }


def normalize_writing_mode(mode: Optional[str]) -> str:
    mode = str(mode or WRITING_MODE_AUTO).strip().lower().replace("-", "_")
    return mode if mode in WRITING_MODES else WRITING_MODE_AUTO


def normalize_fit_mode(mode: Optional[str]) -> str:
    mode = str(mode or FIT_MODE_SHRINK).strip().lower().replace("-", "_")
    return mode if mode in FIT_MODES else FIT_MODE_SHRINK


def normalize_line_break_strategy(strategy: Optional[str]) -> str:
    strategy = str(strategy or LINE_BREAK_AUTO).strip().lower().replace("-", "_")
    return strategy if strategy in LINE_BREAK_STRATEGIES else LINE_BREAK_AUTO


def contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def contains_korean(text: str) -> bool:
    return bool(KOREAN_RE.search(text or ""))


def contains_rtl(text: str) -> bool:
    return bool(RTL_RE.search(text or ""))


def script_bucket(text: str) -> str:
    text = text or ""
    if contains_rtl(text):
        return "rtl"
    if contains_korean(text):
        return "korean"
    if contains_cjk(text):
        return "cjk"
    if any(ord(ch) > 0x1F000 for ch in text):
        return "emoji"
    return "latin"


def resolve_writing_mode(mode: Optional[str], text: str, box_size: Optional[Tuple[float, float]] = None) -> str:
    mode = normalize_writing_mode(mode)
    if mode != WRITING_MODE_AUTO:
        return mode
    if contains_rtl(text):
        return WRITING_MODE_RTL
    w, h = box_size or (0.0, 0.0)
    if contains_cjk(text) and h > max(w * 1.15, 1.0):
        return WRITING_MODE_VERTICAL_RL
    return WRITING_MODE_HORIZONTAL_LTR


def normalize_vertical_punctuation(text: str) -> str:
    out = text or ""
    for src, dst in VERTICAL_PUNCT_MAP.items():
        out = out.replace(src, dst)
    return out


def vertical_punctuation_class(ch: str) -> str:
    if ch in VERTICAL_CENTER_PUNCT:
        return "center"
    if ch in VERTICAL_ROTATE_PUNCT:
        return "rotate"
    cat = unicodedata.category(ch or "")
    if cat.startswith("P"):
        return "punct"
    return "normal"


def _is_cjk_char(ch: str) -> bool:
    return bool(CJK_RE.match(ch or ""))


def kinsoku_wrap(text: str, max_chars: int, strategy: str = LINE_BREAK_AUTO) -> List[str]:
    """Script-aware greedy wrap with selectable manga lettering line-break policy.

    `cjk_strict` keeps Japanese/Chinese closing punctuation off line starts and
    opening punctuation off line ends; `balanced` also avoids one-character
    dangling last lines; `loose` allows simpler wrapping for SFX.
    """
    strategy = normalize_line_break_strategy(strategy)
    text = (text or "").strip()
    if not text:
        return []
    max_chars = max(1, int(max_chars or 1))
    tokens: List[str] = []
    buf = ""
    for ch in text:
        if ch == "\n":
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append("\n")
            continue
        if ch.isspace():
            if buf:
                tokens.append(buf)
                buf = ""
            continue
        if _is_cjk_char(ch) or ch in OPENING_PUNCT or ch in CLOSING_PUNCT:
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        else:
            buf += ch
    if buf:
        tokens.append(buf)

    lines: List[str] = []
    cur = ""
    for tok in tokens:
        if tok == "\n":
            if cur:
                lines.append(cur)
                cur = ""
            continue
        candidate = cur + tok
        if cur and len(candidate) > max_chars:
            if strategy != LINE_BREAK_LOOSE and tok in CLOSING_PUNCT:
                cur += tok
                lines.append(cur)
                cur = ""
                continue
            if strategy != LINE_BREAK_LOOSE and cur[-1:] in OPENING_PUNCT:
                cur += tok
                continue
            lines.append(cur)
            cur = tok
        else:
            cur = candidate
    if cur:
        lines.append(cur)

    for i in range(1, len(lines)):
        while strategy != LINE_BREAK_LOOSE and lines[i] and lines[i][0] in CLOSING_PUNCT and len(lines[i - 1]) < max_chars + 2:
            lines[i - 1] += lines[i][0]
            lines[i] = lines[i][1:]
        if strategy != LINE_BREAK_LOOSE and lines[i - 1].endswith(tuple(OPENING_PUNCT)) and lines[i]:
            lines[i - 1] += lines[i][0]
            lines[i] = lines[i][1:]
    lines = [ln for ln in lines if ln]
    if strategy == LINE_BREAK_BALANCED and len(lines) > 1 and len(lines[-1]) == 1 and len(lines[-2]) > 2:
        lines[-1] = lines[-2][-1] + lines[-1]
        lines[-2] = lines[-2][:-1]
    return lines


def balance_lines(text: str, max_chars: int, strategy: str = LINE_BREAK_BALANCED) -> str:
    lines = kinsoku_wrap(text, max_chars, strategy)
    if len(lines) <= 2:
        return "\n".join(lines) if lines else (text or "")
    total = sum(len(ln) for ln in lines)
    target = max(1, int(math.ceil(total / len(lines))))
    return "\n".join(kinsoku_wrap(text, min(max_chars, max(target, max_chars // 2)), strategy))


def vertical_columns(text: str, max_chars_per_column: int, strategy: str = LINE_BREAK_CJK_STRICT) -> List[str]:
    """Return logical vertical-rl columns. Each string is top-to-bottom; list order is right-to-left."""
    normalized = normalize_vertical_punctuation(text).replace("\n", "")
    return kinsoku_wrap(normalized, max_chars_per_column, strategy)


def estimate_text_bounds(
    text: str,
    font_size: float,
    mode: str = WRITING_MODE_HORIZONTAL_LTR,
    max_width: float = 0.0,
    max_height: float = 0.0,
    line_spacing: float = 1.15,
    letter_spacing: float = 1.0,
    padding: float = 0.0,
    stroke_width: float = 0.0,
    shadow_radius: float = 0.0,
    shadow_offset: Sequence[float] | None = None,
    line_break_strategy: str = LINE_BREAK_AUTO,
) -> Tuple[float, float, int, int]:
    """Cheap renderer-independent bounds estimate including stroke/shadow/padding in pixels."""
    text = text or ""
    fs = max(1.0, float(font_size or 1.0))
    mode = normalize_writing_mode(mode)
    line_break_strategy = normalize_line_break_strategy(line_break_strategy)
    avg = fs * 0.56 * max(0.1, float(letter_spacing or 1.0))
    cjk_avg = fs * max(0.1, float(letter_spacing or 1.0))
    leading = fs * max(0.1, float(line_spacing or 1.0))
    if mode == WRITING_MODE_VERTICAL_RL:
        usable_h = max(1.0, float(max_height or fs * 8) - 2 * padding)
        max_chars = max(1, int(usable_h / max(1.0, cjk_avg)))
        cols = vertical_columns(text, max_chars, line_break_strategy)
        measured_w = max(1, len(cols)) * leading
        measured_h = max((len(c) for c in cols), default=0) * cjk_avg
        line_count = max((len(c) for c in cols), default=0)
        col_count = len(cols)
    else:
        usable_w = max(1.0, float(max_width or fs * 12) - 2 * padding)
        max_chars = max(1, int(usable_w / max(1.0, avg)))
        if mode == WRITING_MODE_RTL:
            lines = kinsoku_wrap(text, max_chars, line_break_strategy)
        elif contains_cjk(text):
            lines = kinsoku_wrap(text, max_chars, line_break_strategy)
        else:
            raw = text.splitlines() or [text]
            lines = []
            for para in raw:
                words = para.split()
                cur = ""
                for word in words or [para]:
                    cand = word if not cur else cur + " " + word
                    if cur and len(cand) > max_chars:
                        lines.append(cur)
                        cur = word
                    else:
                        cur = cand
                if cur:
                    lines.append(cur)
        measured_w = max((len(ln) for ln in lines), default=0) * avg
        measured_h = max(1, len(lines)) * leading
        line_count = len(lines)
        col_count = 0
    sw_px = fs * max(0.0, float(stroke_width or 0.0))
    shadow_offset = shadow_offset or (0.0, 0.0)
    shadow_px = fs * max(0.0, float(shadow_radius or 0.0)) + fs * max(abs(float(shadow_offset[0] if len(shadow_offset) > 0 else 0.0)), abs(float(shadow_offset[1] if len(shadow_offset) > 1 else 0.0)))
    extra = 2 * padding + 2 * sw_px + shadow_px
    return measured_w + extra, measured_h + extra, line_count, col_count


def fit_font_size_to_box(
    text: str,
    font_size: float,
    box_size: Tuple[float, float],
    fit_mode: str = FIT_MODE_SHRINK,
    writing_mode: str = WRITING_MODE_AUTO,
    min_font_size: float = 6.0,
    max_font_size: float = 96.0,
    line_spacing: float = 1.15,
    letter_spacing: float = 1.0,
    padding: float = 0.0,
    stroke_width: float = 0.0,
    line_break_strategy: str = LINE_BREAK_AUTO,
) -> Tuple[float, str, TextRenderDiagnostics]:
    fit_mode = normalize_fit_mode(fit_mode)
    line_break_strategy = normalize_line_break_strategy(line_break_strategy)
    box_w, box_h = box_size or (0.0, 0.0)
    resolved = resolve_writing_mode(writing_mode, text, box_size)
    text_out = text or ""
    if fit_mode == FIT_MODE_BALANCE and resolved != WRITING_MODE_VERTICAL_RL:
        avg = max(1.0, max(1.0, font_size) * 0.56 * max(0.1, letter_spacing))
        text_out = balance_lines(text_out.replace("\n", ""), max(2, int(max(1.0, box_w - 2 * padding) / avg)), line_break_strategy)
    if resolved == WRITING_MODE_VERTICAL_RL:
        text_out = normalize_vertical_punctuation(text_out)
    lo = max(1.0, float(min_font_size or 1.0))
    hi = max(lo, float(max_font_size or 96.0))
    current = max(lo, min(hi, float(font_size or lo)))

    def over(size: float) -> Tuple[bool, Tuple[float, float, int, int]]:
        b = estimate_text_bounds(text_out, size, resolved, box_w, box_h, line_spacing, letter_spacing, padding, stroke_width, line_break_strategy=line_break_strategy)
        return b[0] > box_w or b[1] > box_h, b

    if fit_mode == FIT_MODE_PRESERVE:
        is_over, bounds = over(current)
        diag = TextRenderDiagnostics(resolved, is_over, (bounds[0], bounds[1]), (box_w, box_h), [], "", fit_mode, current, bounds[2], bounds[3], line_break_strategy)
        return current, text_out, diag

    target_largest = fit_mode in (FIT_MODE_EXPAND, FIT_MODE_BALANCE)
    low, high = lo, hi if target_largest else current
    best = current
    for _ in range(14):
        mid = (low + high) / 2.0
        is_over, _ = over(mid)
        if is_over:
            high = mid
        else:
            best = mid
            low = mid
    if fit_mode == FIT_MODE_SHRINK:
        best = min(current, best)
    bounds = estimate_text_bounds(text_out, best, resolved, box_w, box_h, line_spacing, letter_spacing, padding, stroke_width, line_break_strategy=line_break_strategy)
    overflow = bounds[0] > box_w or bounds[1] > box_h
    diag = TextRenderDiagnostics(resolved, overflow, (bounds[0], bounds[1]), (box_w, box_h), [], "", fit_mode, best, bounds[2], bounds[3], line_break_strategy)
    return best, text_out, diag



def _qt_font_metrics_available() -> bool:
    # Headless Linux CI images may have qtpy installed but lack libGL; avoid
    # importing QtGui in that environment so diagnostics degrade gracefully.
    return not (sys.platform.startswith("linux") and ctypes.util.find_library("GL") is None)


def font_supports_char(font_family: str, ch: str) -> bool:
    """Return whether a Qt font family can render a character."""
    if not ch or ch.isspace():
        return True
    if ch in {"□", "�"}:
        return False
    if not _qt_font_metrics_available():
        return True
    from qtpy.QtGui import QFont, QFontMetrics

    metrics = QFontMetrics(QFont(font_family or ""))
    if hasattr(metrics, "inFontUcs4"):
        return bool(metrics.inFontUcs4(ord(ch)))
    if hasattr(metrics, "inFont"):
        return bool(metrics.inFont(ch))
    return True


def choose_font_for_char(ch: str, families: Sequence[str]) -> str:
    """Choose the first family in a fallback chain that can render ch."""
    if not families:
        return ""
    if not ch or ch.isspace():
        return families[0]
    for family in families:
        if font_supports_char(family, ch):
            return family
    return families[0]


def font_fallback_runs(text: str, primary_family: str, config_obj=None, override_chain: str = "") -> List[Tuple[int, int, str]]:
    """Build contiguous fallback font runs for characters unsupported by the primary font.

    Returned runs are `(start, end, family)` offsets suitable for QTextCursor
    selection. Primary-family runs are omitted because the document default font
    already covers them.
    """
    text = text or ""
    families = merge_font_fallback_chain(primary_family, text, config_obj, override_chain)
    if len(families) <= 1:
        return []
    primary = families[0]
    runs: List[Tuple[int, int, str]] = []
    start = None
    active_family = ""
    for idx, ch in enumerate(text):
        family = primary if ch.isspace() or font_supports_char(primary, ch) else choose_font_for_char(ch, families[1:])
        if family == primary:
            if start is not None:
                runs.append((start, idx, active_family))
                start = None
                active_family = ""
            continue
        if start is None:
            start = idx
            active_family = family
        elif family != active_family:
            runs.append((start, idx, active_family))
            start = idx
            active_family = family
    if start is not None:
        runs.append((start, len(text), active_family))
    return runs


def missing_glyphs_after_fallback(primary_family: str, text: str, config_obj=None, override_chain: str = "", limit: int = 8) -> List[str]:
    """Characters that no font in the merged fallback chain appears to render."""
    families = merge_font_fallback_chain(primary_family, text, config_obj, override_chain)
    missing: List[str] = []
    for ch in text or "":
        if ch.isspace() or ch in missing:
            continue
        if not any(font_supports_char(family, ch) for family in families):
            missing.append(ch)
            if len(missing) >= limit:
                break
    return missing

def first_missing_glyphs(font_family: str, text: str, limit: int = 8) -> List[str]:
    """Best-effort glyph diagnostic using Qt when available; safe fallback otherwise."""
    if not text:
        return []
    missing: List[str] = []
    if not _qt_font_metrics_available():
        return [ch for ch in text if ch in {"□", "�"}][:limit]
    from qtpy.QtGui import QFont, QFontMetrics

    font = QFont(font_family or "")
    metrics = QFontMetrics(font)
    for ch in text:
        if ch.isspace() or ch in missing:
            continue
        ok = True
        if hasattr(metrics, "inFontUcs4"):
            ok = bool(metrics.inFontUcs4(ord(ch)))
        elif hasattr(metrics, "inFont"):
            ok = bool(metrics.inFont(ch))
        if not ok:
            missing.append(ch)
            if len(missing) >= limit:
                break
    if not missing:
        missing = [ch for ch in text if ch in {"□", "�"}][:limit]
    return missing


def fallback_chain_for_text(text: str, config_obj=None) -> str:
    bucket = script_bucket(text)
    if config_obj is None:
        return ""
    if bucket == "rtl":
        return getattr(config_obj, "render_fallback_fonts_rtl", "")
    if bucket == "korean":
        return getattr(config_obj, "render_fallback_fonts_korean", "")
    if bucket == "cjk":
        return getattr(config_obj, "render_fallback_fonts_cjk", "")
    if bucket == "emoji":
        return getattr(config_obj, "render_fallback_fonts_emoji", "")
    return getattr(config_obj, "render_fallback_fonts_latin", "")


def merge_font_fallback_chain(primary_family: str, text: str, config_obj=None, override_chain: str = "") -> List[str]:
    families: List[str] = []
    for chunk in [primary_family, override_chain, fallback_chain_for_text(text, config_obj)]:
        for family in str(chunk or "").split(','):
            fam = family.strip()
            if fam and fam not in families:
                families.append(fam)
    return families
