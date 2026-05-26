"""
Theme Engine — token-based QSS with programmatic palette generation.

Wraps and extends the existing `parse_stylesheet` system with:
  - Programmatic color palette (tints/shades) from a primary color
  - Semantic token management
  - Preset primary colors
  - `theme_changed` signal for live updates
  - DPI-aware size scaling

Inspired by Comic Translate's `dayu_widgets.MTheme`.
"""

import json
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union

from qtpy.QtCore import QObject, Signal
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication, QWidget

from utils import shared as C
from utils.logger import LOGGER


# Preset primary colors (name → hex)
PRESET_PRIMARY_COLORS = {
    "blue": "#1E93E5",
    "purple": "#9C4DCC",
    "pink": "#E91E63",
    "red": "#F44336",
    "orange": "#FF9800",
    "green": "#4CAF50",
    "teal": "#009688",
    "cyan": "#00BCD4",
    "indigo": "#3F51B5",
    "yellow": "#FFEB3B",
}


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Linear interpolate between two RGB colors. t in [0, 1]."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def generate_palette(
    primary_hex: str,
    background_hex: str = "#1E1E1E",
    surface_hex: str = "#252526",
    is_dark: bool = True,
) -> Dict[str, str]:
    """
    Generate a full color palette from a primary color.

    Returns a dict of token → value strings (hex or rgba) suitable for QSS substitution.
    """
    primary = _hex_to_rgb(primary_hex)
    bg = _hex_to_rgb(background_hex)
    surface = _hex_to_rgb(surface_hex)

    # Text colors
    if is_dark:
        text_primary = (255, 255, 255)
        text_secondary = (180, 180, 180)
        text_disabled = (120, 120, 120)
    else:
        text_primary = (30, 30, 30)
        text_secondary = (100, 100, 100)
        text_disabled = (180, 180, 180)

    # Generate tints (mix with white) and shades (mix with black)
    tints = [_lerp_color(primary, (255, 255, 255), i / 10) for i in range(1, 11)]
    shades = [_lerp_color(primary, (0, 0, 0), i / 10) for i in range(1, 11)]

    r, g, b = primary
    tokens: Dict[str, str] = {
        "@primaryColor": primary_hex,
        "@primaryColorRgb": f"{r}, {g}, {b}",
        "@primaryColorRgba20": f"rgba({r}, {g}, {b}, 0.2)",
        "@primaryColorRgba35": f"rgba({r}, {g}, {b}, 0.35)",
        "@primaryColorRgba80": f"rgba({r}, {g}, {b}, 0.8)",
        "@primaryLight": _rgb_to_hex(*tints[3]),
        "@primaryDark": _rgb_to_hex(*shades[2]),
        "@backgroundColor": background_hex,
        "@backgroundColorRgb": f"{bg[0]}, {bg[1]}, {bg[2]}",
        "@surfaceColor": surface_hex,
        "@surfaceColorRgb": f"{surface[0]}, {surface[1]}, {surface[2]}",
        "@textPrimaryColor": _rgb_to_hex(*text_primary),
        "@textSecondaryColor": _rgb_to_hex(*text_secondary),
        "@textDisabledColor": _rgb_to_hex(*text_disabled),
        "@borderColor": _rgb_to_hex(*_lerp_color(bg, surface, 0.5)),
        "@hoverColor": _rgb_to_hex(*_lerp_color(surface, primary, 0.08)),
        "@selectedColor": _rgb_to_hex(*_lerp_color(surface, primary, 0.15)),
        "@pressedColor": _rgb_to_hex(*_lerp_color(surface, primary, 0.25)),
    }

    # Accent tints (for gradients, hover states)
    for i, tint in enumerate(tints[:5], start=1):
        tokens[f"@primaryTint{i}"] = _rgb_to_hex(*tint)
    for i, shade in enumerate(shades[:5], start=1):
        tokens[f"@primaryShade{i}"] = _rgb_to_hex(*shade)

    return tokens


class ThemeEngine(QObject):
    """
    Singleton theme manager that generates palettes, applies QSS, and notifies widgets.

    Usage:
        engine = ThemeEngine.instance()
        engine.theme_changed.connect(my_widget.on_theme_changed)
        engine.set_primary_color("#E91E63")
        engine.apply_theme()
    """

    theme_changed = Signal(str)  # Emits the current theme name
    palette_changed = Signal()  # Emits when color palette updates

    _instance: Optional["ThemeEngine"] = None

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._primary_color: str = PRESET_PRIMARY_COLORS["blue"]
        self._is_dark: bool = True
        self._theme_name: str = "eva-dark"
        self._custom_tokens: Dict[str, str] = {}
        self._cached_stylesheet: Optional[str] = None

    @classmethod
    def instance(cls) -> "ThemeEngine":
        if cls._instance is None:
            cls._instance = ThemeEngine()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    # --- Configuration ---

    def set_primary_color(self, color: Union[str, QColor]) -> None:
        """Set the primary accent color (hex string or QColor)."""
        if isinstance(color, QColor):
            color = color.name()
        self._primary_color = color
        self._cached_stylesheet = None

    def set_dark_mode(self, dark: bool) -> None:
        """Toggle dark/light mode."""
        self._is_dark = dark
        self._theme_name = "eva-dark" if dark else "eva-light"
        self._cached_stylesheet = None

    def set_theme_name(self, name: str) -> None:
        """Set the theme name key (must exist in themes.json)."""
        self._theme_name = name
        self._cached_stylesheet = None

    def set_custom_token(self, token: str, value: str) -> None:
        """Override a single QSS token (e.g. '@myColor', '#FF0000')."""
        self._custom_tokens[token] = value
        self._cached_stylesheet = None

    def set_custom_tokens(self, tokens: Dict[str, str]) -> None:
        """Batch-set custom tokens."""
        self._custom_tokens.update(tokens)
        self._cached_stylesheet = None

    def remove_custom_token(self, token: str) -> None:
        self._custom_tokens.pop(token, None)
        self._cached_stylesheet = None

    # --- Palette generation ---

    def generate_tokens(self) -> Dict[str, str]:
        """
        Build the complete token dictionary for QSS substitution.
        Combines generated palette, theme JSON values, and custom overrides.
        """
        bg = "#1E1E1E" if self._is_dark else "#F5F5F5"
        surface = "#252526" if self._is_dark else "#FFFFFF"

        tokens = generate_palette(self._primary_color, bg, surface, self._is_dark)

        # Merge with existing theme JSON (backwards compatibility)
        theme_dict = self._load_theme_dict()
        if self._theme_name in theme_dict:
            for k, v in theme_dict[self._theme_name].items():
                if k not in tokens:
                    tokens[k] = v

        # Apply custom overrides last (highest priority)
        tokens.update(self._custom_tokens)
        return tokens

    def _load_theme_dict(self) -> Dict[str, Dict[str, str]]:
        """Load the themes.json file."""
        try:
            if osp.isfile(C.THEME_PATH):
                with open(C.THEME_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            LOGGER.warning("ThemeEngine: failed to load %s: %s", C.THEME_PATH, e)
        return {}

    # --- Stylesheet compilation ---

    def compile_stylesheet(self, reverse_icon: bool = False) -> str:
        """
        Compile the final QSS string by reading the base stylesheet and substituting tokens.
        """
        if self._cached_stylesheet is not None:
            return self._cached_stylesheet

        # Read base stylesheet
        if not osp.isfile(C.STYLESHEET_PATH):
            LOGGER.error("ThemeEngine: stylesheet not found at %s", C.STYLESHEET_PATH)
            return ""

        with open(C.STYLESHEET_PATH, "r", encoding="utf-8") as f:
            stylesheet = f.read()

        # Token substitution (longest keys first to avoid partial replacements)
        tokens = self.generate_tokens()
        sorted_tokens = sorted(tokens.items(), key=lambda kv: len(kv[0]), reverse=True)
        for key, val in sorted_tokens:
            stylesheet = stylesheet.replace(key, val)

        # Icon fallback
        _icons_dir = osp.join(C.PROGRAM_PATH, "icons")
        if not osp.exists(osp.join(_icons_dir, "drawingtools_texteraser.svg")):
            stylesheet = stylesheet.replace("drawingtools_texteraser.svg", "drawingtools_pen.svg")

        self._cached_stylesheet = stylesheet
        return stylesheet

    def apply_theme(self, reverse_icon: bool = False, widgets: Optional[List[QWidget]] = None) -> None:
        """
        Apply the compiled stylesheet to the application or specific widgets.
        Emits `theme_changed` and `palette_changed` after application.
        """
        stylesheet = self.compile_stylesheet(reverse_icon=reverse_icon)

        if widgets is None:
            app = QApplication.instance()
            if app is not None:
                app.setStyleSheet(stylesheet)
        else:
            for w in widgets:
                w.setStyleSheet(stylesheet)

        # Update shared constants for icon tinting
        try:
            tokens = self.generate_tokens()
            fg_hex = tokens.get("@textPrimaryColor", "#FFFFFF" if self._is_dark else "#1E1E1E")
            C.FOREGROUND_FONTCOLOR = _hex_to_rgb(fg_hex)
        except Exception:
            pass

        self.palette_changed.emit()
        self.theme_changed.emit(self._theme_name)

    def apply_to_widget(self, widget: QWidget) -> None:
        """Apply the current stylesheet to a single widget (useful for popups)."""
        widget.setStyleSheet(self.compile_stylesheet())

    # --- Palette query ---

    def color(self, token: str) -> QColor:
        """Return a QColor for a given token name (e.g. '@primaryColor')."""
        tokens = self.generate_tokens()
        val = tokens.get(token, "#000000")
        return QColor(val)

    def is_dark(self) -> bool:
        return self._is_dark

    def theme_name(self) -> str:
        return self._theme_name

    def primary_color(self) -> str:
        return self._primary_color

    # --- Presets ---

    @staticmethod
    def preset_colors() -> Dict[str, str]:
        """Return available preset primary color names and hex values."""
        return PRESET_PRIMARY_COLORS.copy()

    def set_preset(self, name: str) -> bool:
        """Set primary color by preset name. Returns False if name unknown."""
        if name in PRESET_PRIMARY_COLORS:
            self.set_primary_color(PRESET_PRIMARY_COLORS[name])
            return True
        return False

    # --- DPI scaling ---

    @staticmethod
    def dpi_scale(value: int) -> int:
        """Scale a pixel value by the application's DPI factor."""
        app = QApplication.instance()
        if app is None:
            return value
        try:
            dpi = app.primaryScreen().logicalDotsPerInch()
            return int(value * dpi / 96.0)
        except Exception:
            return value
