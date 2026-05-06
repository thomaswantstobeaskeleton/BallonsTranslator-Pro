from __future__ import annotations

import math
import re
import sys
import unicodedata
import ctypes.util
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict


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
    overflow_axes: List[str] = None
    recommended_actions: List[str] = None

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
            "overflow_axes": list(self.overflow_axes or []),
            "recommended_actions": list(self.recommended_actions or []),
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


def line_break_opportunities(text: str, strategy: str = LINE_BREAK_AUTO) -> List[Dict[str, object]]:
    """Return script-aware break opportunities for diagnostics and layout review.

    The helper is intentionally renderer-independent: UI/API callers can explain
    why a line was rebalanced without needing QTextLayout. `allowed=False`
    captures basic kinsoku bans such as starting a line with closing punctuation
    or ending a line after opening punctuation.
    """
    strategy = normalize_line_break_strategy(strategy)
    text = text or ""
    opportunities: List[Dict[str, object]] = []
    for idx in range(1, len(text)):
        prev_ch = text[idx - 1]
        next_ch = text[idx]
        allowed = True
        reason = "default"
        if strategy != LINE_BREAK_LOOSE and next_ch in CLOSING_PUNCT:
            allowed = False
            reason = "kinsoku_no_line_start_closing_punctuation"
        elif strategy != LINE_BREAK_LOOSE and prev_ch in OPENING_PUNCT:
            allowed = False
            reason = "kinsoku_no_line_end_opening_punctuation"
        elif _is_cjk_char(prev_ch) or _is_cjk_char(next_ch):
            reason = "cjk_character_boundary"
        elif prev_ch.isspace() or next_ch.isspace():
            reason = "word_boundary"
        opportunities.append({"index": idx, "allowed": allowed, "reason": reason, "before": prev_ch, "after": next_ch})
    return opportunities



def optimal_kinsoku_wrap(text: str, max_chars: int, strategy: str = LINE_BREAK_BALANCED) -> List[str]:
    """Dynamic-programming wrap that balances manga lettering while honoring kinsoku.

    The greedy wrapper is still used for fast/default layout. This DP helper is
    used for deliberate balance/fix passes where avoiding a one-character final
    line and reducing ragged columns matter more than speed. It works on
    characters for CJK text and on existing word chunks for Latin/RTL text, then
    rejects breaks that would start a line with closing punctuation or end a
    line after opening punctuation.
    """
    strategy = normalize_line_break_strategy(strategy)
    text = (text or "").strip()
    if not text:
        return []
    max_chars = max(1, int(max_chars or 1))
    if "\n" in text:
        out: List[str] = []
        for para in text.splitlines():
            out.extend(optimal_kinsoku_wrap(para, max_chars, strategy))
        return [ln for ln in out if ln]
    if strategy == LINE_BREAK_LOOSE or len(text) <= max_chars:
        return kinsoku_wrap(text, max_chars, strategy)

    # Tokenize similarly to kinsoku_wrap, but split Latin words only at spaces.
    tokens: List[str] = []
    buf = ""
    for ch in text:
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
    n = len(tokens)
    if n == 0:
        return []

    lengths = [len(tok) for tok in tokens]
    prefix = [0]
    for ln in lengths:
        prefix.append(prefix[-1] + ln)

    def chunk(i: int, j: int) -> str:
        return "".join(tokens[i:j])

    def valid_break(i: int, j: int) -> bool:
        if i >= j:
            return False
        line = chunk(i, j)
        if len(line) > max_chars + 2:
            return False
        if strategy != LINE_BREAK_LOOSE:
            if line[0] in CLOSING_PUNCT:
                return False
            if line[-1] in OPENING_PUNCT:
                return False
        return True

    target = max(1.0, min(float(max_chars), max(1.0, len(text) / max(1, math.ceil(len(text) / max_chars)))))
    dp = [float("inf")] * (n + 1)
    nxt = [-1] * (n + 1)
    dp[n] = 0.0
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n + 1):
            ln = prefix[j] - prefix[i]
            if ln > max_chars + 2:
                break
            if not valid_break(i, j):
                continue
            dangling_penalty = 25.0 if (j == n and ln == 1 and i > 0) else 0.0
            overflow_penalty = 100.0 * max(0, ln - max_chars)
            ragged_penalty = (target - min(target, float(ln))) ** 2
            score = ragged_penalty + dangling_penalty + overflow_penalty + dp[j]
            if score < dp[i]:
                dp[i] = score
                nxt[i] = j
    if nxt[0] < 0:
        return kinsoku_wrap(text, max_chars, strategy)
    lines: List[str] = []
    i = 0
    while i < n and nxt[i] > i:
        j = nxt[i]
        lines.append(chunk(i, j))
        i = j
    return [ln for ln in lines if ln]

def balance_lines(text: str, max_chars: int, strategy: str = LINE_BREAK_BALANCED) -> str:
    strategy = normalize_line_break_strategy(strategy)
    if strategy == LINE_BREAK_BALANCED:
        lines = optimal_kinsoku_wrap(text, max_chars, strategy)
    else:
        lines = kinsoku_wrap(text, max_chars, strategy)
    if len(lines) <= 2:
        return "\n".join(lines) if lines else (text or "")
    total = sum(len(ln) for ln in lines)
    target = max(1, int(math.ceil(total / len(lines))))
    return "\n".join(optimal_kinsoku_wrap(text, min(max_chars, max(target, max_chars // 2)), strategy))


def vertical_columns(text: str, max_chars_per_column: int, strategy: str = LINE_BREAK_CJK_STRICT) -> List[str]:
    """Return logical vertical-rl columns. Each string is top-to-bottom; list order is right-to-left."""
    normalized = normalize_vertical_punctuation(text).replace("\n", "")
    return kinsoku_wrap(normalized, max_chars_per_column, strategy)




def vertical_layout_plan(
    text: str,
    max_chars_per_column: int,
    font_size: float = 24.0,
    line_spacing: float = 1.1,
    letter_spacing: float = 1.0,
    strategy: str = LINE_BREAK_CJK_STRICT,
) -> Dict[str, object]:
    """Build a renderer-independent vertical-RL glyph placement plan.

    Columns are ordered right-to-left and glyphs top-to-bottom. Punctuation
    records a placement class so the Qt renderer, layout review agent, PSD
    handoff, and automation API can agree on what needs centering/rotation/hang
    without using a rotate-horizontal fallback.
    """
    fs = max(1.0, float(font_size or 1.0))
    col_advance = fs * max(0.1, float(line_spacing or 1.0))
    glyph_advance = fs * max(0.1, float(letter_spacing or 1.0))
    cols = vertical_columns(text, max_chars_per_column, strategy)
    glyphs: List[Dict[str, object]] = []
    for col_idx, col in enumerate(cols):
        x = -col_idx * col_advance
        for row_idx, ch in enumerate(col):
            cls = vertical_punctuation_class(ch)
            glyphs.append({
                "char": ch,
                "column": col_idx,
                "row": row_idx,
                "x": x,
                "y": row_idx * glyph_advance,
                "punctuation_class": cls,
                "center": cls in {"center", "punct"},
                "rotate": cls == "rotate",
                "hang": cls == "center" and ch in {"、", "。", "，", "．"},
            })
    return {
        "columns": cols,
        "glyphs": glyphs,
        "column_count": len(cols),
        "max_rows": max((len(c) for c in cols), default=0),
        "column_advance": col_advance,
        "glyph_advance": glyph_advance,
        "strategy": normalize_line_break_strategy(strategy),
    }


def relative_luminance(rgb: Sequence[float]) -> float:
    vals = []
    for c in list(rgb or [0, 0, 0])[:3]:
        c = max(0.0, min(255.0, float(c))) / 255.0
        vals.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    while len(vals) < 3:
        vals.append(0.0)
    return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]


def contrast_ratio(rgb_a: Sequence[float], rgb_b: Sequence[float]) -> float:
    l1 = relative_luminance(rgb_a)
    l2 = relative_luminance(rgb_b)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def suggest_manga_effects_for_background(
    fill_rgb: Sequence[float],
    background_rgb: Sequence[float],
    current_stroke_width: float = 0.0,
    min_ratio: float = 4.5,
) -> Dict[str, object]:
    """Suggest conservative lettering effects when text blends into a bubble/page."""
    ratio = contrast_ratio(fill_rgb, background_rgb)
    fill_lum = relative_luminance(fill_rgb)
    bg_lum = relative_luminance(background_rgb)
    stroke_rgb = [0, 0, 0] if fill_lum > bg_lum else [255, 255, 255]
    stroke_width = max(float(current_stroke_width or 0.0), 0.06 if ratio < min_ratio else float(current_stroke_width or 0.0))
    return {
        "contrast_ratio": round(ratio, 2),
        "needs_effect": ratio < min_ratio,
        "recommended_stroke_rgb": stroke_rgb,
        "recommended_stroke_width": stroke_width,
        "recommended_shadow_radius": 0.04 if ratio < min_ratio else 0.0,
        "reason": "low_text_background_contrast" if ratio < min_ratio else "contrast_ok",
    }


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
        axes = []
        if bounds[0] > box_w:
            axes.append("x")
        if bounds[1] > box_h:
            axes.append("y")
        actions = ["shrink_to_fit"] if axes else []
        diag = TextRenderDiagnostics(resolved, is_over, (bounds[0], bounds[1]), (box_w, box_h), [], "", fit_mode, current, bounds[2], bounds[3], line_break_strategy, axes, actions)
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
    axes = []
    if bounds[0] > box_w:
        axes.append("x")
    if bounds[1] > box_h:
        axes.append("y")
    actions = []
    if overflow:
        actions.append("shrink_to_fit")
    if fit_mode == FIT_MODE_BALANCE or (overflow and resolved != WRITING_MODE_VERTICAL_RL):
        actions.append("balance_lines")
    if resolved == WRITING_MODE_VERTICAL_RL:
        actions.append("check_vertical_punctuation")
    diag = TextRenderDiagnostics(resolved, overflow, (bounds[0], bounds[1]), (box_w, box_h), [], "", fit_mode, best, bounds[2], bounds[3], line_break_strategy, axes, actions)
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
