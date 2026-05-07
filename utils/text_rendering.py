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

READING_ORDER_AUTO = "auto"
READING_ORDER_RTL = "rtl"
READING_ORDER_LTR = "ltr"
READING_ORDER_TTB = "ttb"
READING_ORDERS = {READING_ORDER_AUTO, READING_ORDER_RTL, READING_ORDER_LTR, READING_ORDER_TTB}

CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
KOREAN_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
RTL_RE = re.compile(r"[\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
OPENING_PUNCT = set("([{（［｛〈《「『【〔〖〝‘“")
CLOSING_PUNCT = set(")]},.!?:;、。，．！？：；⁈⁉‼⁇…）］｝〉》」』】〕〗〟’”")
# Basic kinsoku shori sets for manga lettering.  These extend generic
# punctuation bans with Japanese small kana, iteration/prolongation marks,
# and dash-like marks that should not be stranded at the start/end of a
# balanced line.
KINSOKU_PROHIBITED_LINE_START = CLOSING_PUNCT | set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヶヵゝゞ々ーｰ゛゜ヽヾ・゠〰〜～")
KINSOKU_PROHIBITED_LINE_END = OPENING_PUNCT | set("〔〖〘〚〝‘“")
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
VERTICAL_OPEN_TO_CLOSE = {"（": "）", "［": "］", "｛": "｝", "〈": "〉", "《": "》", "「": "」", "『": "』", "【": "】", "〔": "〕", "(": ")", "[": "]", "{": "}"}

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

STYLE_PRESET_FIELDS = (
    "font_family",
    "font_size",
    "font_weight",
    "bold",
    "italic",
    "frgb",
    "srgb",
    "stroke_width",
    "shadow_radius",
    "shadow_strength",
    "shadow_color",
    "shadow_offset",
    "line_spacing",
    "letter_spacing",
    "alignment",
    "writing_mode",
    "fit_mode",
    "line_break_strategy",
    "text_padding",
    "fit_font_size_min",
    "fit_font_size_max",
    "opacity",
    "fallback_font_chain",
)


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
    safe_inner_bounds: Tuple[float, float] = (0.0, 0.0)
    effect_margin: float = 0.0
    quality_score: float = 1.0
    recommended_box_size: Tuple[float, float] = (0.0, 0.0)
    box_scale_hint: float = 1.0
    ink_clip_risk: bool = False
    preset_suggestion: str = ""

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
            "safe_inner_bounds": list(self.safe_inner_bounds),
            "effect_margin": self.effect_margin,
            "quality_score": self.quality_score,
            "recommended_box_size": list(self.recommended_box_size),
            "box_scale_hint": self.box_scale_hint,
            "ink_clip_risk": bool(self.ink_clip_risk),
            "preset_suggestion": self.preset_suggestion,
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


def normalize_reading_order(order: Optional[str]) -> str:
    order = str(order or READING_ORDER_AUTO).strip().lower().replace("-", "_")
    aliases = {"right_to_left": READING_ORDER_RTL, "left_to_right": READING_ORDER_LTR, "top_to_bottom": READING_ORDER_TTB}
    order = aliases.get(order, order)
    return order if order in READING_ORDERS else READING_ORDER_AUTO


def preset_id_from_label(label: str, existing: Optional[Sequence[str]] = None) -> str:
    """Return a stable custom preset id from a user-facing label."""
    base = re.sub(r"[^a-z0-9]+", "_", (label or "").strip().lower()).strip("_") or "custom_preset"
    existing_set = {str(x) for x in (existing or [])}
    candidate = f"custom:{base}"
    n = 2
    while candidate in existing_set or candidate in MANGA_PRESETS:
        candidate = f"custom:{base}_{n}"
        n += 1
    return candidate


def sanitize_manga_preset(preset: Dict[str, object], label: str = "") -> Dict[str, object]:
    """Clamp a user-saved manga lettering preset to supported, serializable fields."""
    src = dict(preset or {})
    out: Dict[str, object] = {"label": str(src.get("label") or label or "Custom preset").strip() or "Custom preset"}
    for key in STYLE_PRESET_FIELDS:
        if key not in src:
            continue
        value = src.get(key)
        if key in {"writing_mode", "fit_mode", "line_break_strategy"}:
            if key == "writing_mode":
                value = normalize_writing_mode(value)
            elif key == "fit_mode":
                value = normalize_fit_mode(value)
            else:
                value = normalize_line_break_strategy(value)
        elif key in {"font_size", "stroke_width", "shadow_radius", "shadow_strength", "line_spacing", "letter_spacing", "text_padding", "fit_font_size_min", "fit_font_size_max", "opacity"}:
            try:
                value = float(value)
            except Exception:
                continue
            if key == "opacity":
                value = max(0.0, min(1.0, value))
            elif key == "font_size":
                value = max(1.0, min(512.0, value))
            else:
                value = max(0.0, value)
        elif key in {"bold", "italic"}:
            value = bool(value)
        elif key in {"alignment"}:
            try:
                value = max(0, min(2, int(value)))
            except Exception:
                continue
        elif key in {"frgb", "srgb", "shadow_color", "shadow_offset"}:
            try:
                seq = list(value or [])
                limit = 2 if key == "shadow_offset" else 3
                value = [float(v) if key == "shadow_offset" else int(max(0, min(255, int(v)))) for v in seq[:limit]]
                while len(value) < limit:
                    value.append(0.0 if key == "shadow_offset" else 0)
            except Exception:
                continue
        elif key == "font_weight":
            if value is None:
                continue
            try:
                value = int(value)
            except Exception:
                continue
        else:
            value = str(value or "").strip()
        out[key] = value
    return out


def custom_manga_presets(config_obj=None) -> Dict[str, Dict[str, object]]:
    raw = getattr(config_obj, "render_custom_manga_presets", {}) if config_obj is not None else {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for preset_id, preset in raw.items():
        pid = str(preset_id or "").strip()
        if not pid:
            continue
        if not pid.startswith("custom:"):
            pid = "custom:" + pid
        out[pid] = sanitize_manga_preset(preset, label=pid.split(":", 1)[-1].replace("_", " "))
    return out


def manga_presets(config_obj=None) -> Dict[str, Dict[str, object]]:
    """Built-in presets plus persisted user presets for UI/API/batch code."""
    presets = {k: dict(v) for k, v in MANGA_PRESETS.items()}
    presets.update(custom_manga_presets(config_obj))
    return presets


def preset_from_font_format(font_format, label: str = "Custom preset") -> Dict[str, object]:
    values: Dict[str, object] = {"label": label}
    for key in STYLE_PRESET_FIELDS:
        if hasattr(font_format, key):
            value = getattr(font_format, key)
            if isinstance(value, (list, tuple)):
                value = list(value)
            values[key] = value
    return sanitize_manga_preset(values, label=label)


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



def infer_reading_order(blocks: Sequence[object], default: str = READING_ORDER_AUTO) -> str:
    """Infer manga page reading order from textbox geometry and script hints.

    RTL manga pages commonly have right-to-left columns; webtoon/caption-heavy
    pages often read top-to-bottom. This is intentionally conservative and can
    be overridden by config/API callers.
    """
    order = normalize_reading_order(default)
    if order != READING_ORDER_AUTO:
        return order
    boxes = []
    cjk = rtl = 0
    for blk in blocks or []:
        xyxy = getattr(blk, "xyxy", None) or [0, 0, 0, 0]
        if len(xyxy) >= 4:
            x1, y1, x2, y2 = [float(v or 0.0) for v in xyxy[:4]]
            boxes.append((x1, y1, x2, y2))
        text = str(getattr(blk, "translation", "") or getattr(blk, "rich_text", "") or " ".join(getattr(blk, "text", []) or []))
        if contains_rtl(text):
            rtl += 1
        if contains_cjk(text):
            cjk += 1
    if rtl:
        return READING_ORDER_RTL
    if len(boxes) >= 3:
        page_w = max((b[2] for b in boxes), default=0.0) - min((b[0] for b in boxes), default=0.0)
        page_h = max((b[3] for b in boxes), default=0.0) - min((b[1] for b in boxes), default=0.0)
        if page_h > page_w * 1.8:
            return READING_ORDER_TTB
    return READING_ORDER_RTL if cjk else READING_ORDER_LTR


def reading_order_key(block: object, order: str = READING_ORDER_AUTO, row_tolerance: float = 40.0) -> Tuple[float, float, float]:
    xyxy = getattr(block, "xyxy", None) or [0, 0, 0, 0]
    x1, y1, x2, y2 = [float(v or 0.0) for v in (list(xyxy) + [0, 0, 0, 0])[:4]]
    row = round(y1 / max(1.0, float(row_tolerance or 1.0)))
    order = normalize_reading_order(order)
    if order == READING_ORDER_RTL:
        return (row, -x1, y1)
    if order == READING_ORDER_LTR:
        return (row, x1, y1)
    if order == READING_ORDER_TTB:
        return (y1, x1, -x2)
    return (row, x1, y1)


def sort_blocks_for_reading_order(blocks: Sequence[object], order: str = READING_ORDER_AUTO, row_tolerance: float = 40.0) -> Tuple[List[object], str]:
    resolved = infer_reading_order(blocks, order)
    return sorted(list(blocks or []), key=lambda b: reading_order_key(b, resolved, row_tolerance)), resolved

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


def locale_aware_upper(text: str, locale: str = "") -> str:
    """Uppercase text without damaging scripts that have no useful uppercase form.

    Koharu issue #572/#567 focused on universal uppercase behavior. Python's
    `str.upper()` is acceptable for Latin/Greek/Cyrillic, but applying it to
    manga text containing Japanese/Chinese/Korean, Arabic, Hebrew, or emoji can
    change punctuation expectations without improving lettering. This helper
    uppercases only contiguous cased-script runs and keeps other scripts as-is.
    Turkish/Azeri dotted-i is handled explicitly when requested.
    """
    text = text or ""
    loc = (locale or "").lower()
    turkic = loc.startswith(("tr", "az"))
    out: List[str] = []
    for ch in text:
        if not ch:
            continue
        cat = unicodedata.category(ch)
        name = unicodedata.name(ch, "")
        if turkic and ch == "i":
            out.append("İ")
        elif turkic and ch == "ı":
            out.append("I")
        elif cat.startswith("L") and not (_is_cjk_char(ch) or KOREAN_RE.match(ch) or RTL_RE.match(ch)) and any(script in name for script in ("LATIN", "GREEK", "CYRILLIC")):
            out.append(ch.upper())
        else:
            out.append(ch)
    return "".join(out)


def suggest_manga_preset(text: str, box_size: Tuple[float, float] = (0.0, 0.0), writing_mode: str = WRITING_MODE_AUTO) -> str:
    """Suggest a lettering preset from script, geometry, and SFX-like content."""
    text = text or ""
    w, h = box_size or (0.0, 0.0)
    resolved = resolve_writing_mode(writing_mode, text, box_size)
    stripped = re.sub(r"\s+", "", text)
    emph = sum(1 for ch in stripped if ch in "!！?？⁈⁉‼⁇~～ー—―")
    if resolved == WRITING_MODE_VERTICAL_RL:
        return "vertical_cjk_bubble"
    if len(stripped) <= 8 and (emph >= 2 or stripped.isupper()):
        return "sfx_bold"
    if w > 0 and h > 0 and w > h * 2.5:
        return "caption_box"
    if len(stripped) <= 12 and max(w, h) < 140:
        return "small_aside"
    return "default_manga_bubble"


def ink_clip_risk(box_size: Tuple[float, float], measured_bounds: Tuple[float, float], effect_margin: float, threshold_px: float = 1.0) -> bool:
    """True when stroke/shadow/padding likely clips at the text box edge."""
    bw, bh = box_size or (0.0, 0.0)
    mw, mh = measured_bounds or (0.0, 0.0)
    spare_x = float(bw or 0.0) - float(mw or 0.0)
    spare_y = float(bh or 0.0) - float(mh or 0.0)
    return effect_margin > 0 and (spare_x < max(threshold_px, effect_margin * 0.3) or spare_y < max(threshold_px, effect_margin * 0.3))


def normalize_vertical_punctuation(text: str) -> str:
    out = text or ""
    for src, dst in VERTICAL_PUNCT_MAP.items():
        out = out.replace(src, dst)
    return out


def glyph_advance_units(text: str, mode: str = WRITING_MODE_HORIZONTAL_LTR) -> float:
    """Estimate visual advance units more accurately than raw len().

    CJK lettering has full-width ideographs but many punctuation marks occupy
    half/centered cells, while emoji/symbols often need a wider cell. This
    dependency-free estimate improves fit diagnostics and layout review without
    requiring platform font metrics.
    """
    mode = normalize_writing_mode(mode)
    total = 0.0
    for ch in text or "":
        if ch == "\n":
            continue
        if ch.isspace():
            total += 0.35
        elif ch in VERTICAL_CENTER_PUNCT or unicodedata.category(ch).startswith("P"):
            total += 0.55 if mode != WRITING_MODE_VERTICAL_RL else 0.80
        elif _is_cjk_char(ch) or KOREAN_RE.match(ch):
            total += 1.0
        elif ord(ch) >= 0x1F000:
            total += 1.2
        elif unicodedata.combining(ch):
            total += 0.0
        else:
            total += 0.56
    return total


def max_visual_line_units(lines: Sequence[str], mode: str = WRITING_MODE_HORIZONTAL_LTR) -> float:
    return max((glyph_advance_units(line, mode) for line in lines or []), default=0.0)


def recommended_tight_letter_spacing(current: float, overflow_ratio: float, floor: float = 0.88) -> float:
    """Return a conservative manga-lettering tracking value for overflow fixes."""
    cur = max(0.1, float(current or 1.0))
    if overflow_ratio <= 1.0:
        return cur
    # Do not over-condense; leave shrinking/resizing to the next review action.
    return round(max(float(floor), min(cur, cur / min(1.12, overflow_ratio))), 3)


def vertical_tate_chu_yoko_groups(text: str) -> List[Dict[str, object]]:
    """Find short ASCII digit/Latin punctuation runs that should stay upright in vertical text.

    This is a renderer-neutral tate-chu-yoko hint: the current Qt renderer can
    consume it when available, while QA/PSD/API paths can already expose the
    intent. Groups are intentionally limited to 2 characters, matching common
    manga lettering for dates, issue numbers, and compact !?/!! marks.
    """
    groups: List[Dict[str, object]] = []
    text = normalize_vertical_punctuation(text or "")
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isascii() and (ch.isdigit() or ch.isalpha() or ch in "!?+-"):
            j = i + 1
            while j < len(text) and j - i < 2 and text[j].isascii() and (text[j].isdigit() or text[j].isalpha() or text[j] in "!?+-"):
                j += 1
            if j - i >= 2:
                groups.append({"start": i, "end": j, "text": text[i:j], "orientation": "upright_compact"})
                i = j
                continue
        i += 1
    return groups


def effect_margin_px(font_size: float, stroke_width: float = 0.0, shadow_radius: float = 0.0, shadow_offset: Sequence[float] | None = None, padding: float = 0.0) -> float:
    fs = max(1.0, float(font_size or 1.0))
    shadow_offset = shadow_offset or (0.0, 0.0)
    sx = abs(float(shadow_offset[0] if len(shadow_offset) > 0 else 0.0))
    sy = abs(float(shadow_offset[1] if len(shadow_offset) > 1 else 0.0))
    return max(0.0, float(padding or 0.0)) + fs * max(0.0, float(stroke_width or 0.0)) + fs * max(0.0, float(shadow_radius or 0.0)) + fs * max(sx, sy)


def safe_inner_bounds(box_size: Tuple[float, float], font_size: float, stroke_width: float = 0.0, shadow_radius: float = 0.0, shadow_offset: Sequence[float] | None = None, padding: float = 0.0) -> Tuple[Tuple[float, float], float]:
    """Return effect-aware inner width/height and total margin for diagnostics."""
    w, h = box_size or (0.0, 0.0)
    margin = effect_margin_px(font_size, stroke_width, shadow_radius, shadow_offset, padding)
    return (max(1.0, float(w or 0.0) - 2 * margin), max(1.0, float(h or 0.0) - 2 * margin)), margin


def lettering_quality_score(overflow: bool, missing_glyphs: Sequence[str] = (), contrast_ratio_value: float = 7.0, effect_margin: float = 0.0, box_size: Tuple[float, float] = (0.0, 0.0)) -> float:
    """Small 0..1 score used by QA/API to sort lettering issues."""
    score = 1.0
    if overflow:
        score -= 0.35
    if missing_glyphs:
        score -= min(0.30, 0.08 * len(set(missing_glyphs)))
    if contrast_ratio_value < 4.5:
        score -= min(0.25, (4.5 - contrast_ratio_value) / 10.0)
    min_side = max(1.0, min(float(box_size[0] or 0.0), float(box_size[1] or 0.0)))
    if effect_margin > min_side * 0.18:
        score -= 0.15
    return round(max(0.0, min(1.0, score)), 3)



def vertical_punctuation_adjustment(ch: str, font_size: float = 24.0) -> Dict[str, float | bool | str]:
    """Return renderer-neutral positioning hints for vertical punctuation.

    Qt text items can use these values for custom painting later, while QA, PSD
    handoff, and tests can already reason about punctuation that should be
    centered, rotated, or hung inside vertical CJK columns.
    """
    cls = vertical_punctuation_class(ch)
    fs = max(1.0, float(font_size or 1.0))
    out: Dict[str, float | bool | str] = {"class": cls, "dx": 0.0, "dy": 0.0, "rotate_degrees": 0.0, "scale": 1.0, "hang": False}
    if cls == "center":
        out.update({"dx": -0.12 * fs, "dy": -0.08 * fs})
        if ch in {"、", "。", "，", "．"}:
            out.update({"dx": 0.18 * fs, "dy": -0.18 * fs, "hang": True})
    elif cls == "rotate":
        out.update({"rotate_degrees": 90.0})
        if ch in VERTICAL_OPEN_TO_CLOSE or ch in set(VERTICAL_OPEN_TO_CLOSE.values()):
            out.update({"dy": -0.05 * fs, "scale": 0.96})
    return out


def vertical_bracket_pair_hints(text: str) -> List[Dict[str, int | str]]:
    """Pair simple Japanese/Chinese brackets for vertical-layout diagnostics."""
    stack: List[Tuple[str, int]] = []
    pairs: List[Dict[str, int | str]] = []
    close_to_open = {v: k for k, v in VERTICAL_OPEN_TO_CLOSE.items()}
    for idx, ch in enumerate(normalize_vertical_punctuation(text or "")):
        if ch in VERTICAL_OPEN_TO_CLOSE:
            stack.append((ch, idx))
        elif ch in close_to_open:
            want = close_to_open[ch]
            for pos in range(len(stack) - 1, -1, -1):
                open_ch, open_idx = stack[pos]
                if open_ch == want:
                    pairs.append({"open_index": open_idx, "close_index": idx, "open": open_ch, "close": ch})
                    del stack[pos:]
                    break
    return pairs

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
            if strategy != LINE_BREAK_LOOSE and tok in KINSOKU_PROHIBITED_LINE_START:
                cur += tok
                lines.append(cur)
                cur = ""
                continue
            if strategy != LINE_BREAK_LOOSE and cur[-1:] in KINSOKU_PROHIBITED_LINE_END:
                cur += tok
                continue
            lines.append(cur)
            cur = tok
        else:
            cur = candidate
    if cur:
        lines.append(cur)

    for i in range(1, len(lines)):
        while strategy != LINE_BREAK_LOOSE and lines[i] and lines[i][0] in KINSOKU_PROHIBITED_LINE_START and len(lines[i - 1]) < max_chars + 2:
            lines[i - 1] += lines[i][0]
            lines[i] = lines[i][1:]
        if strategy != LINE_BREAK_LOOSE and lines[i - 1].endswith(tuple(KINSOKU_PROHIBITED_LINE_END)) and lines[i]:
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
        if strategy != LINE_BREAK_LOOSE and next_ch in KINSOKU_PROHIBITED_LINE_START:
            allowed = False
            reason = "kinsoku_no_line_start_closing_punctuation"
        elif strategy != LINE_BREAK_LOOSE and prev_ch in KINSOKU_PROHIBITED_LINE_END:
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
            if line[0] in KINSOKU_PROHIBITED_LINE_START:
                return False
            if line[-1] in KINSOKU_PROHIBITED_LINE_END:
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

def split_long_word(word: str, max_chars: int, marker: str = "‐") -> List[str]:
    """Split a single long Latin token at readable boundaries for narrow bubbles.

    This is a dependency-free fallback inspired by Koharu word-splitting requests:
    prefer camelCase, separators, and vowel/consonant-ish boundaries before hard
    splitting. CJK text is handled by character wrapping elsewhere.
    """
    word = (word or "").replace("\u200b", "")
    max_chars = max(2, int(max_chars or 2))
    if len(word) <= max_chars:
        return [word] if word else []
    pieces: List[str] = []
    rest = word
    vowels = set("aeiouAEIOU")
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = -1
        for i in range(len(window) - 1, max(1, max_chars // 2), -1):
            if window[i] in "-_/:\u00ad" or (window[i - 1].islower() and window[i].isupper()) or (window[i - 1] in vowels and window[i] not in vowels):
                cut = i + (0 if window[i] in "-_/:" else 0)
                break
        if cut <= 1:
            cut = max_chars - 1
        chunk = rest[:cut].rstrip("-_/:\u00ad")
        if chunk and not chunk.endswith(marker):
            chunk += marker
        pieces.append(chunk)
        rest = rest[cut:].lstrip("-_/:\u00ad")
    if rest:
        pieces.append(rest)
    return [p for p in pieces if p]


def wrap_latin_text(text: str, max_chars: int, split_long_words: bool = True) -> List[str]:
    lines: List[str] = []
    for para in (text or "").splitlines() or [text or ""]:
        words = para.split()
        cur = ""
        for raw_word in words or [para]:
            word_parts = split_long_word(raw_word, max_chars) if split_long_words and len(raw_word) > max_chars else [raw_word]
            for word in word_parts:
                cand = word if not cur else cur + " " + word
                if cur and len(cand) > max_chars:
                    lines.append(cur)
                    cur = word
                else:
                    cur = cand
        if cur:
            lines.append(cur)
    return [ln for ln in lines if ln]


def resolve_fit_font_size_bounds(style_min: float = 0.0, style_max: float = 0.0, global_min: float = 6.0, global_max: float = 96.0) -> Tuple[float, float]:
    lo = float(style_min or 0.0) if float(style_min or 0.0) > 0 else float(global_min or 6.0)
    hi = float(style_max or 0.0) if float(style_max or 0.0) > 0 else float(global_max or 96.0)
    lo = max(1.0, lo)
    hi = max(lo, hi)
    return lo, hi


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
    rotate_latin: bool = True,
    punctuation_hang: bool = True,
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
    logical_index = 0
    for col_idx, col in enumerate(cols):
        x = -col_idx * col_advance
        for row_idx, ch in enumerate(col):
            cls = vertical_punctuation_class(ch)
            adjust = vertical_punctuation_adjustment(ch, fs)
            glyphs.append({
                "index": logical_index,
                "char": ch,
                "column": col_idx,
                "row": row_idx,
                "x": x,
                "y": row_idx * glyph_advance,
                "punctuation_class": cls,
                "center": cls in {"center", "punct"},
                "rotate": (cls == "rotate") or (rotate_latin and ch.isascii() and ch.isalpha()),
                "hang": bool(punctuation_hang) and (bool(adjust.get("hang")) or (cls == "center" and ch in {"、", "。", "，", "．"})),
                "offset": {"dx": adjust.get("dx", 0.0), "dy": adjust.get("dy", 0.0)},
                "rotate_degrees": adjust.get("rotate_degrees", 0.0),
                "scale": adjust.get("scale", 1.0),
                "tate_chu_yoko": False,
                "tate_chu_yoko_group": -1,
            })
            logical_index += 1
    tcy_groups = vertical_tate_chu_yoko_groups(text)
    for group_idx, group in enumerate(tcy_groups):
        start = int(group.get("start", -1))
        end = int(group.get("end", -1))
        for glyph in glyphs:
            if start <= int(glyph.get("index", -1)) < end:
                glyph["tate_chu_yoko"] = True
                glyph["tate_chu_yoko_group"] = group_idx
    return {
        "columns": cols,
        "glyphs": glyphs,
        "column_count": len(cols),
        "max_rows": max((len(c) for c in cols), default=0),
        "column_advance": col_advance,
        "glyph_advance": glyph_advance,
        "strategy": normalize_line_break_strategy(strategy),
        "tate_chu_yoko_groups": tcy_groups,
        "bracket_pairs": vertical_bracket_pair_hints(text),
        "rotate_latin": bool(rotate_latin),
        "punctuation_hang": bool(punctuation_hang),
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
        measured_h = max((glyph_advance_units(c, mode) for c in cols), default=0.0) * cjk_avg
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
            lines = wrap_latin_text(text, max_chars, split_long_words=True)
        measured_w = max_visual_line_units(lines, mode) * fs * max(0.1, float(letter_spacing or 1.0))
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
    shadow_radius: float = 0.0,
    shadow_offset: Sequence[float] | None = None,
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
        b = estimate_text_bounds(text_out, size, resolved, box_w, box_h, line_spacing, letter_spacing, padding, stroke_width, shadow_radius=shadow_radius, shadow_offset=shadow_offset, line_break_strategy=line_break_strategy)
        return b[0] > box_w or b[1] > box_h, b

    if fit_mode == FIT_MODE_PRESERVE:
        is_over, bounds = over(current)
        axes = []
        if bounds[0] > box_w:
            axes.append("x")
        if bounds[1] > box_h:
            axes.append("y")
        actions = ["shrink_to_fit"] if axes else []
        inner, margin = safe_inner_bounds((box_w, box_h), current, stroke_width=stroke_width, shadow_radius=shadow_radius, shadow_offset=shadow_offset, padding=padding)
        quality = lettering_quality_score(is_over, [], 7.0, margin, (box_w, box_h))
        scale_hint = max(bounds[0] / max(1.0, box_w), bounds[1] / max(1.0, box_h), 1.0)
        rec_box = (round(max(box_w, bounds[0] + 2 * margin), 2), round(max(box_h, bounds[1] + 2 * margin), 2))
        clip = ink_clip_risk((box_w, box_h), (bounds[0], bounds[1]), margin)
        if clip and "increase_padding" not in actions:
            actions.append("increase_padding")
        if axes and float(letter_spacing or 1.0) > 0.9:
            actions.append("tighten_letter_spacing")
        preset = suggest_manga_preset(text_out, (box_w, box_h), resolved)
        diag = TextRenderDiagnostics(resolved, is_over, (bounds[0], bounds[1]), (box_w, box_h), [], "", fit_mode, current, bounds[2], bounds[3], line_break_strategy, axes, actions, inner, margin, quality, rec_box, round(scale_hint, 3), clip, preset)
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
    bounds = estimate_text_bounds(text_out, best, resolved, box_w, box_h, line_spacing, letter_spacing, padding, stroke_width, shadow_radius=shadow_radius, shadow_offset=shadow_offset, line_break_strategy=line_break_strategy)
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
    inner, margin = safe_inner_bounds((box_w, box_h), best, stroke_width=stroke_width, shadow_radius=shadow_radius, shadow_offset=shadow_offset, padding=padding)
    quality = lettering_quality_score(overflow, [], 7.0, margin, (box_w, box_h))
    scale_hint = max(bounds[0] / max(1.0, box_w), bounds[1] / max(1.0, box_h), 1.0)
    rec_box = (round(max(box_w, bounds[0] + 2 * margin), 2), round(max(box_h, bounds[1] + 2 * margin), 2))
    if overflow and "resize_to_fit_content" not in actions:
        actions.append("resize_to_fit_content")
    clip = ink_clip_risk((box_w, box_h), (bounds[0], bounds[1]), margin)
    if clip and "increase_padding" not in actions:
        actions.append("increase_padding")
    if axes and float(letter_spacing or 1.0) > 0.9:
        actions.append("tighten_letter_spacing")
    preset = suggest_manga_preset(text_out, (box_w, box_h), resolved)
    diag = TextRenderDiagnostics(resolved, overflow, (bounds[0], bounds[1]), (box_w, box_h), [], "", fit_mode, best, bounds[2], bounds[3], line_break_strategy, axes, actions, inner, margin, quality, rec_box, round(scale_hint, 3), clip, preset)
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
