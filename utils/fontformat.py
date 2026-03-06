from typing import Union
import enum
import re
import copy

import numpy as np

from . import shared
from .structures import Tuple, Union, List, Dict, Config, field, nested_dataclass


def pt2px(pt, to_int=False) -> float:
    if to_int:
        return int(round(pt * shared.LDPI / 72.))
    else:
        return pt * shared.LDPI / 72.

def px2pt(px) -> float:
    return px / shared.LDPI * 72.


class LineSpacingType(enum.IntEnum):
    Proportional = 0
    Distance = 1


class TextAlignment(enum.IntEnum):
    Left = 0
    Center = 1
    Right = 2


fontweight_qt5_to_qt6 = {0: 100, 12: 200, 25: 300, 50: 400, 57: 500, 63: 600, 75: 700, 81: 800, 87: 900}
fontweight_qt6_to_qt5 = {100: 0, 200: 12, 300: 25, 400: 50, 500: 57, 600: 63, 700: 75, 800: 81, 900: 87}

fontweight_pattern = re.compile(r'font-weight:(\d+)', re.DOTALL)

def fix_fontweight_qt(weight: Union[str, int]):

    def _fix_html_fntweight(matched):
        weight = int(matched.group(1))
        return f'font-weight:{fix_fontweight_qt(weight)}'

    if weight is None:
        return None
    if isinstance(weight, int):
        if shared.FLAG_QT6 and weight < 100:
            if weight in fontweight_qt5_to_qt6:
                weight = fontweight_qt5_to_qt6[weight]
        if not shared.FLAG_QT6 and weight >= 100:
            if weight in fontweight_qt6_to_qt5:
                weight = fontweight_qt6_to_qt5[weight]
    if isinstance(weight, str):
        weight = fontweight_pattern.sub(lambda matched: _fix_html_fntweight(matched), weight)
    return weight


@nested_dataclass
class FontFormat(Config):

    font_family: str = shared.DEFAULT_FONT_FAMILY # to always apply shared.DEFAULT_FONT_FAMILY
    font_size: float = 24
    stroke_width: float = 0.
    frgb: List = field(default_factory=lambda: [0, 0, 0])
    srgb: List = field(default_factory=lambda: [0, 0, 0])
    bold: bool = False
    underline: bool = False
    strikethrough: bool = False
    italic: bool = False
    alignment: int = 0
    vertical: bool = False
    font_weight: int = None
    line_spacing: float = 1.2
    letter_spacing: float = 1.15
    opacity: float = 1.
    shadow_radius: float = 0.
    shadow_strength: float = 1.
    shadow_color: List = field(default_factory=lambda: [0, 0, 0])
    shadow_offset: List = field(default_factory=lambda: [0., 0.])
    gradient_enabled: bool = False
    gradient_type: int = 0  # 0 = Linear, 1 = Radial
    gradient_start_color: List = field(default_factory=lambda: [0, 0, 0])
    gradient_end_color: List = field(default_factory=lambda: [255, 255, 255])
    gradient_angle: float = 0.
    gradient_size: float = 1.0
    _style_name: str = ''
    line_spacing_type: int = LineSpacingType.Proportional
    # Text on path (circular/arc) for manga bubbles and SFX (#1138)
    text_on_path: int = 0  # 0 = none, 1 = circular, 2 = arc
    text_on_path_arc_degrees: float = 180.0  # for arc: span in degrees (e.g. 180 = semicircle)
    # Mesh warp presets (#1093): Arc, Arch, Bulge, Flag (Photoshop-like)
    warp_style: int = 0  # 0 = none, 1 = arc, 2 = arch, 3 = bulge, 4 = flag
    warp_strength: float = 0.5  # 0..1
    # Blend mode for compositing block with scene (0=Normal, 1=Multiply, 2=Screen, 3=Overlay, 4=Darken, 5=Lighten)
    blend_mode: int = 0
    # Auto fit font size to block: when True, layout scales font so text fits the block's bounding box (#1077)
    auto_fit_font_size: bool = False
    # Outline only: draw stroke, no fill (stroke-only text)
    outline_only: bool = False
    # Stroke outline outside only (EasyScanlate-style): draw stroke at 2x width so fill covers inner half; visible stroke is only outside glyphs
    stroke_outline_outside_only: bool = False
    # Overlay image opacity (0..1) and block skew (shear) - synced to TextBlock
    overlay_opacity: float = 1.0
    skew_x: float = 0.0
    skew_y: float = 0.0
    # Text box shape: 0 = rectangle, >0 = corner radius in px (rounded rect)
    text_box_corner_radius: float = 0.0
    # #35: per-character stroke color (list of [r,g,b], one per character; used for text-on-path)
    stroke_rgb_per_char: List = field(default=None)  # optional

    deprecated_attributes: dict = field(default_factory = lambda: dict())

    @property
    def size_pt(self):
        pt = px2pt(self.font_size)
        return max(1.0, pt) if pt <= 0 else pt

    def __post_init__(self):
        da = self.deprecated_attributes
        if len(da) > 0:
            if 'size' in da:
                self.font_size = pt2px(da['size'])
            if 'weight' in da:
                self.font_weight = da['weight']
            if 'family' in da:
                self.font_family = da['family']

        self.font_weight = fix_fontweight_qt(self.font_weight)
        self.deprecated_attributes = {}

    def deepcopy(self):
        fmt_copyed: FontFormat = None
        fmt_copyed = copy.deepcopy(self)
        return fmt_copyed

    def merge(self, target: Config, compare: bool = False):
        if id(self) == id(target):
            return set()
        tgt_keys = target.annotations_set()
        updated_keys = set()
        for key in tgt_keys:
            if not hasattr(self, key):
                continue
            if compare:
                if key != '_style_name':
                    if isinstance(target[key], np.ndarray):
                        is_diff = np.any(self[key] != target[key])
                    else:
                        is_diff = self[key] != target[key]
                    if is_diff:
                        self.update(key, copy.deepcopy(target[key]))
                        updated_keys.add(key)
            else:
                self.update(key, copy.deepcopy(target[key]))
        return updated_keys

    def foreground_color(self):
        return [int(round(x)) for x in self.frgb]
    
    def stroke_color(self):
        return [int(round(x)) for x in self.srgb]