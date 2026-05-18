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
    if (balloon_shape or "auto") in ("round", "diamond", "square", "bevel", "pentagon") and target > 1:
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
