"""
Centralized theme tokens for the BallonsTranslator-Pro shell.

All QML, QSS, and Python colour/spacing constants live here so
every part of the new UI stays consistent with the mockup.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ThemeColors:
    """Dark-purple palette extracted from the mockup."""
    # Backgrounds
    bg_deepest: str = "#121218"       # window chrome / sidebar bg
    bg_base: str = "#1A1A24"          # main content area
    bg_surface: str = "#222233"       # cards, panels
    bg_surface_hover: str = "#2A2A3D" # hovered card
    bg_surface_active: str = "#33334A" # selected / active card
    bg_elevated: str = "#252538"       # modals, drawers
    bg_input: str = "#1E1E2E"          # text inputs

    # Borders
    border: str = "#33334A"
    border_subtle: str = "#2A2A3C"
    border_focus: str = "#7B5CFF"

    # Text
    text_primary: str = "#E8E8F0"
    text_secondary: str = "#A0A0B8"
    text_muted: str = "#6A6A80"
    text_inverse: str = "#121218"

    # Accent – primary purple
    accent: str = "#7B5CFF"
    accent_hover: str = "#9B80FF"
    accent_muted: str = "rgba(123, 92, 255, 0.18)"

    # Accent – secondary (blue used for links / info)
    accent_blue: str = "#3B82F6"
    accent_blue_hover: str = "#60A5FA"

    # Semantic
    success: str = "#4ADE80"
    warning: str = "#FBBF24"
    error: str = "#F87171"
    info: str = "#38BDF8"

    # Progress bar
    progress_start: str = "#7B5CFF"
    progress_end: str = "#5B3FDF"

    # Status pills
    status_running: str = "#7B5CFF"
    status_queued: str = "#6A6A80"
    status_done: str = "#4ADE80"
    status_error: str = "#F87171"

    # Scrollbar
    scrollbar_bg: str = "rgba(0, 0, 0, 30)"
    scrollbar_handle: str = "rgba(123, 92, 255, 0.35)"
    scrollbar_handle_hover: str = "rgba(123, 92, 255, 0.55)"


@dataclass(frozen=True)
class ThemeSpacing:
    """Consistent spacing scale (px)."""
    xs: int = 4
    sm: int = 8
    md: int = 12
    lg: int = 16
    xl: int = 24
    xxl: int = 32


@dataclass(frozen=True)
class ThemeFonts:
    """Font families and sizes."""
    family: str = "Inter"
    family_fallback: str = "Segoe UI"
    family_mono: str = "JetBrains Mono"
    family_mono_fallback: str = "Consolas"

    size_xs: int = 10
    size_sm: int = 11
    size_base: int = 13
    size_lg: int = 15
    size_xl: int = 18
    size_h2: int = 22
    size_h1: int = 28


@dataclass(frozen=True)
class ThemeRadius:
    """Border-radius scale."""
    sm: int = 4
    md: int = 8
    lg: int = 12
    xl: int = 16
    pill: int = 999


@dataclass(frozen=True)
class ThemeShadows:
    """Box-shadow definitions (CSS-like, used in QSS where possible)."""
    sm: str = "0px 1px 3px rgba(0,0,0,0.3)"
    md: str = "0px 4px 12px rgba(0,0,0,0.4)"
    lg: str = "0px 8px 24px rgba(0,0,0,0.5)"


# ── Singleton instances ───────────────────────────────────────
COLORS = ThemeColors()
SPACING = ThemeSpacing()
FONTS = ThemeFonts()
RADIUS = ThemeRadius()
SHADOWS = ThemeShadows()


# ── Sidebar constants ──────────────────────────────────────────
SIDEBAR_WIDTH = 200          # collapsed: icons only; expanded: icons + labels
SIDEBAR_WIDTH_COLLAPSED = 56
TITLEBAR_HEIGHT = 40


# ── Helper: build a full-app QSS string ───────────────────────
def build_shell_stylesheet() -> str:
    """Return a QSS stylesheet that themes all QWidget children of the shell."""
    c = COLORS
    f = FONTS
    r = RADIUS
    return f"""
/* ── Global ─────────────────────────────────────────── */
QWidget {{
    background-color: {c.bg_base};
    color: {c.text_primary};
    font-family: "{f.family}", "{f.family_fallback}", sans-serif;
    font-size: {f.size_base}px;
}}
QWidget:disabled {{
    color: {c.text_muted};
}}

/* ── Scroll bars ────────────────────────────────────── */
QScrollBar:vertical {{
    background: {c.scrollbar_bg};
    width: 8px;
    border-radius: 4px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {c.scrollbar_handle};
    min-height: 30px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{
    background: {c.scrollbar_handle_hover};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {c.scrollbar_bg};
    height: 8px;
    border-radius: 4px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {c.scrollbar_handle};
    min-width: 30px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {c.scrollbar_handle_hover};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── QPushButton ────────────────────────────────────── */
QPushButton {{
    background-color: {c.bg_surface};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {r.md}px;
    padding: 6px 16px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {c.bg_surface_hover};
    border-color: {c.border_focus};
}}
QPushButton:pressed {{
    background-color: {c.bg_surface_active};
}}
QPushButton[accent="true"] {{
    background-color: {c.accent};
    color: {c.text_inverse};
    border: none;
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{
    background-color: {c.accent_hover};
}}

/* ── QLineEdit / QPlainTextEdit / QTextEdit ─────────── */
QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser {{
    background-color: {c.bg_input};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {r.sm}px;
    padding: 4px 8px;
    selection-background-color: {c.accent_muted};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {c.border_focus};
}}

/* ── QComboBox ──────────────────────────────────────── */
QComboBox {{
    background-color: {c.bg_input};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {r.sm}px;
    padding: 4px 8px;
}}
QComboBox:hover {{
    border-color: {c.border_focus};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {c.bg_elevated};
    color: {c.text_primary};
    border: 1px solid {c.border};
    selection-background-color: {c.accent_muted};
}}

/* ── QTabBar ────────────────────────────────────────── */
QTabBar::tab {{
    background-color: {c.bg_surface};
    color: {c.text_secondary};
    border: none;
    padding: 8px 16px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    color: {c.text_primary};
    border-bottom: 2px solid {c.accent};
}}
QTabBar::tab:hover:!selected {{
    color: {c.text_primary};
    background-color: {c.bg_surface_hover};
}}

/* ── QGroupBox ──────────────────────────────────────── */
QGroupBox {{
    background-color: {c.bg_surface};
    border: 1px solid {c.border};
    border-radius: {r.md}px;
    margin-top: 14px;
    padding-top: 16px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    color: {c.text_secondary};
    font-weight: 600;
}}

/* ── QLabel ─────────────────────────────────────────── */
QLabel {{
    background: transparent;
}}

/* ── QSpinBox / QDoubleSpinBox ──────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background-color: {c.bg_input};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {r.sm}px;
    padding: 2px 6px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {c.border_focus};
}}

/* ── QCheckBox / QRadioButton ──────────────────────── */
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {c.border};
    border-radius: 3px;
    background-color: {c.bg_input};
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {c.accent};
    border-color: {c.accent};
}}

/* ── QProgressBar ──────────────────────────────────── */
QProgressBar {{
    background-color: {c.bg_surface};
    border: none;
    border-radius: {r.sm}px;
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {c.accent};
    border-radius: {r.sm}px;
}}

/* ── QToolTip ──────────────────────────────────────── */
QToolTip {{
    background-color: {c.bg_elevated};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {r.sm}px;
    padding: 4px 8px;
}}

/* ── QMenu ─────────────────────────────────────────── */
QMenu {{
    background-color: {c.bg_elevated};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {r.md}px;
    padding: 4px 0;
}}
QMenu::item {{
    padding: 6px 24px;
}}
QMenu::item:selected {{
    background-color: {c.accent_muted};
}}
QMenu::separator {{
    height: 1px;
    background: {c.border};
    margin: 4px 8px;
}}

/* ── QHeaderView ───────────────────────────────────── */
QHeaderView::section {{
    background-color: {c.bg_surface};
    color: {c.text_secondary};
    border: none;
    border-bottom: 1px solid {c.border};
    padding: 6px 8px;
    font-weight: 600;
}}

/* ── QTableView / QTreeView / QListView ────────────── */
QTableView, QTreeView, QListView, QListWidget {{
    background-color: {c.bg_base};
    alternate-background-color: {c.bg_surface};
    border: 1px solid {c.border};
    border-radius: {r.sm}px;
    gridline-color: {c.border_subtle};
}}
QTableView::item:selected, QTreeView::item:selected, QListView::item:selected, QListWidget::item:selected {{
    background-color: {c.accent_muted};
    color: {c.text_primary};
}}

/* ── QSplitter ─────────────────────────────────────── */
QSplitter::handle {{
    background-color: {c.border_subtle};
}}
QSplitter::handle:horizontal {{
    width: 2px;
}}
QSplitter::handle:vertical {{
    height: 2px;
}}

/* ── QDockWidget ───────────────────────────────────── */
QDockWidget {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    color: {c.text_primary};
}}
QDockWidget::title {{
    background-color: {c.bg_surface};
    padding: 6px;
    border-bottom: 1px solid {c.border};
}}

/* ── QStatusBar ────────────────────────────────────── */
QStatusBar {{
    background-color: {c.bg_deepest};
    color: {c.text_muted};
    border-top: 1px solid {c.border_subtle};
}}
"""


def build_sidebar_qss() -> str:
    """QSS specifically for the sidebar QWidget fallback."""
    c = COLORS
    return f"""
#SidebarContainer {{
    background-color: {c.bg_deepest};
    border-right: 1px solid {c.border_subtle};
}}
#SidebarButton {{
    background: transparent;
    color: {c.text_muted};
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    text-align: left;
    font-size: 13px;
    font-weight: 500;
}}
#SidebarButton:hover {{
    background-color: {c.bg_surface};
    color: {c.text_primary};
}}
#SidebarButton[active="true"] {{
    background-color: {c.accent_muted};
    color: {c.accent};
    font-weight: 600;
}}
#SidebarLogo {{
    background: transparent;
    color: {c.accent};
    font-size: 15px;
    font-weight: 700;
    padding: 12px 16px;
}}
#SidebarSeparator {{
    background-color: {c.border_subtle};
    max-height: 1px;
    margin: 4px 16px;
}}
"""
