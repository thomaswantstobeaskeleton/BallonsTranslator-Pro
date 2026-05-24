from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

AUTO_LAYOUT_PRESET_BALANCED = "balanced"
AUTO_LAYOUT_PRESET_FIT = "fit"
AUTO_LAYOUT_PRESET_READABLE = "readable"
AUTO_LAYOUT_PRESETS = (
    AUTO_LAYOUT_PRESET_BALANCED,
    AUTO_LAYOUT_PRESET_FIT,
    AUTO_LAYOUT_PRESET_READABLE,
)


AUTO_LAYOUT_PROFILE_DEFAULTS = {
    AUTO_LAYOUT_PRESET_BALANCED: {
        "layout_constrain_to_bubble": True,
        "layout_center_in_bubble_after_autolayout": True,
        "layout_center_in_bubble_min_gap_px": 40.0,
        "layout_check_overflow_after_layout": True,
        "layout_use_mask_safe_area": True,
        "layout_box_size_check_model_id": "",
        "layout_optimal_breaks": True,
        "layout_hyphenation": True,
        "optimize_line_breaks": False,
        "layout_short_line_penalty": 80.0,
        "layout_height_overflow_penalty": 360.0,
        "layout_font_size_min": 8.0,
        "layout_font_size_max": 72.0,
        "layout_font_fit_bubble": True,
        "layout_font_binary_search": True,
        "layout_auto_final_fit_pass": True,
        # Default to a simple rectangular textbox shape. Users can still opt
        # into Auto/Round/Diamond/etc. from advanced settings or the context menu.
        "layout_balloon_shape": "square",
        "layout_balloon_shape_auto_method": "contour_ratio",
        "layout_balloon_shape_model_id": "",
        "layout_min_line_width_px": 80.0,
        "layout_max_line_width_frac_no_bubble": 0.78,
        "layout_stub_penalty_1word": 2000.0,
    },
    AUTO_LAYOUT_PRESET_FIT: {
        "layout_constrain_to_bubble": True,
        "layout_center_in_bubble_after_autolayout": True,
        "layout_center_in_bubble_min_gap_px": 30.0,
        "layout_check_overflow_after_layout": True,
        "layout_use_mask_safe_area": True,
        "layout_box_size_check_model_id": "",
        "layout_optimal_breaks": True,
        "layout_hyphenation": True,
        "optimize_line_breaks": False,
        "layout_short_line_penalty": 120.0,
        "layout_height_overflow_penalty": 900.0,
        "layout_font_size_min": 6.0,
        "layout_font_size_max": 64.0,
        "layout_font_fit_bubble": True,
        "layout_font_binary_search": True,
        "layout_auto_final_fit_pass": True,
        "layout_balloon_shape": "square",
        "layout_balloon_shape_auto_method": "contour_ratio",
        "layout_balloon_shape_model_id": "",
        "layout_min_line_width_px": 70.0,
        "layout_max_line_width_frac_no_bubble": 0.72,
        "layout_stub_penalty_1word": 2600.0,
    },
    AUTO_LAYOUT_PRESET_READABLE: {
        "layout_constrain_to_bubble": False,
        "layout_center_in_bubble_after_autolayout": True,
        "layout_center_in_bubble_min_gap_px": 50.0,
        "layout_check_overflow_after_layout": True,
        "layout_use_mask_safe_area": True,
        "layout_box_size_check_model_id": "",
        "layout_optimal_breaks": True,
        "layout_hyphenation": True,
        "optimize_line_breaks": True,
        "layout_short_line_penalty": 60.0,
        "layout_height_overflow_penalty": 240.0,
        "layout_font_size_min": 10.0,
        "layout_font_size_max": 84.0,
        "layout_font_fit_bubble": True,
        "layout_font_binary_search": True,
        "layout_auto_final_fit_pass": True,
        "layout_balloon_shape": "square",
        "layout_balloon_shape_auto_method": "contour_ratio",
        "layout_balloon_shape_model_id": "",
        "layout_min_line_width_px": 92.0,
        "layout_max_line_width_frac_no_bubble": 0.86,
        "layout_stub_penalty_1word": 1600.0,
    },
}


def auto_layout_profile_defaults(value: str | None) -> dict:
    """Return user-facing preset defaults for all advanced auto-layout knobs.

    This keeps the many legacy tuning values behind one stable profile while
    still allowing expert users to override individual fields after applying it.
    Model-backed checks are intentionally off in these defaults so automatic
    layout remains local, fast, and predictable unless the user opts into a
    model ID explicitly.
    """
    preset = normalize_auto_layout_preset(value)
    return dict(AUTO_LAYOUT_PROFILE_DEFAULTS[preset])


def apply_auto_layout_profile(config_obj, value: str | None) -> dict:
    """Apply a preset's advanced defaults to a module config-like object."""
    preset = normalize_auto_layout_preset(value)
    defaults = auto_layout_profile_defaults(preset)
    if config_obj is not None:
        setattr(config_obj, "layout_auto_preset", preset)
        for key, val in defaults.items():
            setattr(config_obj, key, val)
    return defaults


def auto_layout_profile_summary(value: str | None) -> str:
    preset = normalize_auto_layout_preset(value)
    if preset == AUTO_LAYOUT_PRESET_FIT:
        return "Strict fit: smaller min font, strong overflow penalties, compact line widths, mask-safe area, and simple rectangular text boxes by default."
    if preset == AUTO_LAYOUT_PRESET_READABLE:
        return "Readable: allows roomier boxes/fewer lines and larger font while keeping final overflow checks and simple rectangular text boxes by default."
    return "Balanced: mask-safe geometry, centered bubbles, optimal line breaks, binary font fitting, and simple rectangular text boxes by default."


def _band(value: float, bands: tuple[tuple[float, str], ...], fallback: str) -> str:
    try:
        val = float(value)
    except Exception:
        return fallback
    for limit, label in bands:
        if val <= limit:
            return label
    return fallback


def auto_layout_setting_hints(values: dict | object | None = None) -> dict:
    """Translate advanced numeric auto-layout values into editor-friendly labels.

    The raw values are still used by the layout engine, but the UI can show these
    labels so users understand whether a number currently means strict, balanced,
    or roomy behavior.  `values` may be a dict, dataclass/config object, or None.
    """
    def get(name: str, default):
        if values is None:
            return default
        if isinstance(values, dict):
            return values.get(name, default)
        return getattr(values, name, default)

    min_pt = float(get("layout_font_size_min", 8.0) or 8.0)
    max_pt = float(get("layout_font_size_max", 72.0) or 72.0)
    short_pen = float(get("layout_short_line_penalty", 80.0) or 0.0)
    height_pen = float(get("layout_height_overflow_penalty", 360.0) or 0.0)
    stub_pen = float(get("layout_stub_penalty_1word", 2000.0) or 0.0)
    min_width = float(get("layout_min_line_width_px", 80.0) or 80.0)
    no_bubble = float(get("layout_max_line_width_frac_no_bubble", 0.78) or 0.78)
    center_gap = float(get("layout_center_in_bubble_min_gap_px", 40.0) or 0.0)
    box_model = str(get("layout_box_size_check_model_id", "") or "").strip()
    shape_model = str(get("layout_balloon_shape_model_id", "") or "").strip()
    shape_method = str(get("layout_balloon_shape_auto_method", "contour_ratio") or "contour_ratio")
    shape_value = str(get("layout_balloon_shape", "square") or "square").strip().lower()

    return {
        "font_range": f"{min_pt:g}–{max_pt:g} pt (" + _band(min_pt, ((6.5, "tiny text allowed"), (9.5, "normal manga range")), "large/readable minimum") + ")",
        "short_line_penalty": _band(short_pen, ((40, "very loose"), (90, "balanced"), (150, "strict")), "very strict"),
        "height_overflow_penalty": _band(height_pen, ((250, "roomy / may grow taller"), (550, "balanced"), (950, "strict fit")), "very strict fit"),
        "stub_penalty": _band(stub_pen, ((900, "loose"), (1800, "balanced"), (2600, "strong")), "maximum"),
        "minimum_line_width": _band(min_width, ((72, "compact/narrow"), (95, "balanced"), (125, "roomy")), "very roomy"),
        "no_bubble_width": _band(no_bubble, ((0.70, "compact"), (0.82, "balanced"), (0.92, "wide/readable")), "very wide"),
        "center_gap": "never skip centering" if center_gap <= 0 else f"skip combined bubbles within {center_gap:g}px",
        "box_model": "geometry only" if not box_model else ("built-in CLIP" if box_model == "builtin" else "custom model"),
        "shape_detection": "simple rectangle" if shape_value in ("square", "rectangle", "rect") else ("model-assisted" if shape_model or shape_method.startswith("model") else "model-free contour/aspect"),
    }


def auto_layout_advanced_summary(values: dict | object | None = None) -> str:
    hints = auto_layout_setting_hints(values)
    return (
        f"Font {hints['font_range']}; lines: {hints['short_line_penalty']} short-line, "
        f"{hints['stub_penalty']} one-word, {hints['height_overflow_penalty']} height; "
        f"widths: {hints['minimum_line_width']} in bubbles / {hints['no_bubble_width']} outside; "
        f"centering: {hints['center_gap']}; checks: {hints['box_model']}, {hints['shape_detection']}."
    )


@dataclass(frozen=True)
class AutoLayoutPresetSettings:
    """Small set of user-facing automatic lettering behaviors.

    Values are intentionally deltas/multipliers applied on top of existing
    shape heuristics so old project settings remain meaningful.
    """

    line_width_delta: float = 0.0
    inner_inset_delta: float = 0.0
    font_scale_multiplier: float = 1.0
    mask_min_coverage: float = 0.88
    mask_min_edge_coverage: float = 0.70
    target_fill: float = 0.78
    max_expand_ratio: float = 1.12


def normalize_auto_layout_preset(value: str | None) -> str:
    value = (value or AUTO_LAYOUT_PRESET_BALANCED).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "auto": AUTO_LAYOUT_PRESET_BALANCED,
        "default": AUTO_LAYOUT_PRESET_BALANCED,
        "balanced": AUTO_LAYOUT_PRESET_BALANCED,
        "safe": AUTO_LAYOUT_PRESET_FIT,
        "strict": AUTO_LAYOUT_PRESET_FIT,
        "fit": AUTO_LAYOUT_PRESET_FIT,
        "fit_inside": AUTO_LAYOUT_PRESET_FIT,
        "fit_inside_bubble": AUTO_LAYOUT_PRESET_FIT,
        "readable": AUTO_LAYOUT_PRESET_READABLE,
        "large": AUTO_LAYOUT_PRESET_READABLE,
        "larger_text": AUTO_LAYOUT_PRESET_READABLE,
        "roomy": AUTO_LAYOUT_PRESET_READABLE,
    }
    return aliases.get(value, AUTO_LAYOUT_PRESET_BALANCED)


def auto_layout_preset_settings(value: str | None) -> AutoLayoutPresetSettings:
    preset = normalize_auto_layout_preset(value)
    if preset == AUTO_LAYOUT_PRESET_FIT:
        return AutoLayoutPresetSettings(
            line_width_delta=-0.06,
            inner_inset_delta=-0.04,
            font_scale_multiplier=0.92,
            mask_min_coverage=0.92,
            mask_min_edge_coverage=0.78,
            target_fill=0.72,
            max_expand_ratio=1.04,
        )
    if preset == AUTO_LAYOUT_PRESET_READABLE:
        return AutoLayoutPresetSettings(
            line_width_delta=0.04,
            inner_inset_delta=0.03,
            font_scale_multiplier=1.08,
            mask_min_coverage=0.84,
            mask_min_edge_coverage=0.62,
            target_fill=0.84,
            max_expand_ratio=1.18,
        )
    return AutoLayoutPresetSettings()



def auto_layout_effective_preset(
    configured: str | None,
    text: str = "",
    box_width: float = 0.0,
    box_height: float = 0.0,
    balloon_shape: str = "auto",
    line_count: int | None = None,
) -> str:
    """Choose the effective preset for one bubble.

    An explicit user choice (fit/readable) wins. The default balanced preset is
    intentionally adaptive: dense/tiny/narrow bubbles become safer, while very
    short text in roomy bubbles gets a readability boost.
    """
    preset = normalize_auto_layout_preset(configured)
    if preset != AUTO_LAYOUT_PRESET_BALANCED:
        return preset
    clean_len = len((text or "").replace("\n", "").strip())
    bw = max(1.0, float(box_width or 0.0))
    bh = max(1.0, float(box_height or 0.0))
    area = bw * bh
    density = clean_len / max(1.0, np.sqrt(area))
    aspect = bw / max(1.0, bh)
    shape = (balloon_shape or "auto").lower()
    if clean_len >= 110 or density >= 1.15 or min(bw, bh) < 70 or aspect <= 0.58 or shape in ("diamond", "point"):
        return AUTO_LAYOUT_PRESET_FIT
    if clean_len <= 22 and area >= 10000 and (line_count is None or line_count <= 2) and shape not in ("diamond", "point", "narrow"):
        return AUTO_LAYOUT_PRESET_READABLE
    return AUTO_LAYOUT_PRESET_BALANCED


def auto_rendered_fit_scale(
    content_width: float,
    content_height: float,
    box_width: float,
    box_height: float,
    preset: str | None = None,
    allow_expand: bool = True,
) -> float:
    """Return a conservative scale for a rendered text document inside a box.

    Values below 1 shrink overflow. Values above 1 are limited expansion for
    underfilled boxes, driven by the selected automatic lettering preset.
    """
    cw = max(1.0, float(content_width or 0.0))
    ch = max(1.0, float(content_height or 0.0))
    bw = max(1.0, float(box_width or 0.0))
    bh = max(1.0, float(box_height or 0.0))
    overflow = max(cw / bw, ch / bh)
    if overflow > 1.0:
        return float(np.clip(0.985 / overflow, 0.50, 0.98))
    if not allow_expand:
        return 1.0
    settings = auto_layout_preset_settings(preset)
    fill = max(cw / bw, ch / bh)
    if fill <= 0 or fill >= settings.target_fill:
        return 1.0
    return float(np.clip(settings.target_fill / fill, 1.0, settings.max_expand_ratio))

def estimate_target_line_count(
    text_width: float,
    line_height: int,
    box_width: float,
    box_height: float,
    balloon_shape: str = "auto",
) -> int:
    """Estimate a pleasant line count for automatic comic lettering.

    The result is intentionally heuristic: it gives the width search a set of
    useful candidates instead of relying only on repeated shrinking from the
    widest possible line. Wide/short balloons prefer fewer lines; tall/narrow
    balloons prefer more lines; round/diamond balloons prefer a compact stack.
    """
    if text_width <= 0 or box_width <= 0 or line_height <= 0:
        return 1
    by_width = max(1, int(np.ceil(text_width / max(1.0, box_width * 0.86))))
    by_height = max(1, int(np.floor(max(1.0, box_height) / max(1, line_height)))) if box_height > 0 else by_width + 2
    aspect = box_width / max(1.0, box_height)
    if aspect >= 2.0:
        target = max(1, by_width - 1)
    elif aspect <= 0.72:
        target = by_width + 1
    else:
        target = by_width
    if (balloon_shape or "auto") in ("round", "diamond", "square", "rectangle", "rect", "bevel", "pentagon") and target > 1:
        target += 1
    return int(max(1, min(max(1, by_height + 1), target)))


def candidate_layout_widths(
    max_width: float,
    min_width: float,
    words_width: float,
    delimiter_total_width: float,
    line_height: int,
    target_box_height: int = None,
    balloon_shape: str = "auto",
    optimize_for_fewer_lines: bool = False,
) -> List[int]:
    """Build width candidates for automatic line breaking.

    This combines text length, target height, and bubble shape candidates with
    the legacy geometric shrink series so the scorer can choose a balanced stack
    rather than being limited to near-max-width lines.
    """
    if not np.isfinite(max_width) or max_width <= 0:
        max_width = max(min_width, words_width + delimiter_total_width)
    max_width = max(float(min_width), float(max_width))
    total_width = max(1.0, float(words_width + delimiter_total_width))
    box_height = float(target_box_height or max(line_height, line_height * 3))
    target_lines = estimate_target_line_count(total_width, line_height, max_width, box_height, balloon_shape)
    if optimize_for_fewer_lines:
        target_lines = max(1, target_lines - 1)
    max_lines_by_height = max(1, int(box_height / max(1, line_height))) if box_height > 0 else target_lines + 2

    raw = [max_width]
    for delta in (-2, -1, 0, 1, 2, 3):
        n = max(1, min(max_lines_by_height + 2, target_lines + delta))
        raw.append((total_width / n) * 1.04)
    shape = (balloon_shape or "auto").lower()
    shape_fracs = {
        "round": (0.62, 0.72, 0.82, 0.92),
        "diamond": (0.56, 0.66, 0.78, 0.88),
        "narrow": (0.50, 0.60, 0.72, 0.84),
        "point": (0.52, 0.64, 0.76, 0.88),
        "square": (0.66, 0.76, 0.86, 0.96),
        "rectangle": (0.66, 0.76, 0.86, 0.96),
        "rect": (0.66, 0.76, 0.86, 0.96),
        "bevel": (0.64, 0.74, 0.86, 0.96),
        "pentagon": (0.62, 0.74, 0.84, 0.94),
        "elongated": (0.74, 0.84, 0.92, 1.0),
    }.get(shape, (0.60, 0.72, 0.84, 0.96))
    raw.extend(max_width * f for f in shape_fracs)
    shrink = 0.96 if optimize_for_fewer_lines else 0.97
    raw.extend(max_width * (shrink ** i) for i in range(24))

    out: List[int] = []
    seen = set()
    for val in raw:
        width = int(round(max(float(min_width), min(float(max_width), float(val)))))
        if width not in seen:
            seen.add(width)
            out.append(width)
    return out
