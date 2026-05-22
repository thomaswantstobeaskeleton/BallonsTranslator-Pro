from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SpacingTokens:
    xs: int = 4
    sm: int = 8
    md: int = 12
    lg: int = 16
    xl: int = 24
    xxl: int = 32
    xxxl: int = 48


@dataclass(frozen=True)
class RadiusTokens:
    sm: int = 4
    md: int = 8
    lg: int = 12
    xl: int = 16
    pill: int = 999


@dataclass(frozen=True)
class TypographyTokens:
    caption: int = 11
    body: int = 13
    body_large: int = 14
    title: int = 16
    headline: int = 20
    display: int = 28


@dataclass(frozen=True)
class ColorTokens:
    background: str = "#111318"
    background_soft: str = "#151922"
    surface: str = "#1a1d24"
    surface_alt: str = "#222733"
    surface_hover: str = "#293040"
    surface_glass: str = "rgba(34, 39, 51, 0.78)"
    border: str = "#303645"
    border_soft: str = "rgba(255, 255, 255, 0.08)"
    text: str = "#f2f4f8"
    text_muted: str = "#aab2c0"
    text_subtle: str = "#788295"
    primary: str = "#7aa2ff"
    primary_soft: str = "rgba(122, 162, 255, 0.16)"
    success: str = "#5cc887"
    success_soft: str = "rgba(92, 200, 135, 0.16)"
    warning: str = "#f4c76b"
    warning_soft: str = "rgba(244, 199, 107, 0.18)"
    danger: str = "#ff7b7b"
    danger_soft: str = "rgba(255, 123, 123, 0.16)"
    info: str = "#7fd7ff"
    info_soft: str = "rgba(127, 215, 255, 0.16)"
    dango_pink: str = "#ff9ecf"
    dango_blue: str = "#9fd4ff"
    dango_lavender: str = "#b9a7ff"


@dataclass(frozen=True)
class ShadowTokens:
    sm: str = "0 1px 2px rgba(0, 0, 0, 0.18)"
    md: str = "0 8px 24px rgba(0, 0, 0, 0.24)"
    lg: str = "0 18px 48px rgba(0, 0, 0, 0.32)"


@dataclass(frozen=True)
class MotionTokens:
    fast_ms: int = 120
    normal_ms: int = 180
    slow_ms: int = 260


@dataclass(frozen=True)
class UITokens:
    spacing: SpacingTokens = SpacingTokens()
    radius: RadiusTokens = RadiusTokens()
    typography: TypographyTokens = TypographyTokens()
    colors: ColorTokens = ColorTokens()
    shadows: ShadowTokens = ShadowTokens()
    motion: MotionTokens = MotionTokens()


TOKENS = UITokens()

STATUS_COLORS: Dict[str, str] = {
    "idle": TOKENS.colors.text_muted,
    "running": TOKENS.colors.info,
    "success": TOKENS.colors.success,
    "warning": TOKENS.colors.warning,
    "error": TOKENS.colors.danger,
    "disabled": TOKENS.colors.text_muted,
    "experimental": TOKENS.colors.warning,
    "cat": TOKENS.colors.success,
    "beta": TOKENS.colors.dango_lavender,
}

STATUS_BACKGROUNDS: Dict[str, str] = {
    "idle": "transparent",
    "running": TOKENS.colors.info_soft,
    "success": TOKENS.colors.success_soft,
    "warning": TOKENS.colors.warning_soft,
    "error": TOKENS.colors.danger_soft,
    "disabled": "transparent",
    "experimental": TOKENS.colors.warning_soft,
    "cat": TOKENS.colors.success_soft,
    "beta": "rgba(185, 167, 255, 0.16)",
}

WORKFLOW_ACCENTS: Dict[str, str] = {
    "home": TOKENS.colors.dango_lavender,
    "editor": TOKENS.colors.primary,
    "live": TOKENS.colors.info,
    "quick_image": TOKENS.colors.success,
    "downloader": TOKENS.colors.warning,
    "batch": TOKENS.colors.dango_blue,
    "assist": TOKENS.colors.success,
    "models": TOKENS.colors.primary,
    "settings": TOKENS.colors.text_muted,
    "diagnostics": TOKENS.colors.warning,
}

WORKFLOW_GRADIENTS: Dict[str, str] = {
    "home": "qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(185,167,255,0.30), stop:1 rgba(159,212,255,0.12))",
    "editor": "qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(122,162,255,0.26), stop:1 rgba(127,215,255,0.10))",
    "live": "qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(127,215,255,0.26), stop:1 rgba(255,158,207,0.10))",
    "quick_image": "qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(92,200,135,0.24), stop:1 rgba(127,215,255,0.10))",
    "downloader": "qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(244,199,107,0.24), stop:1 rgba(255,158,207,0.10))",
    "assist": "qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(92,200,135,0.24), stop:1 rgba(185,167,255,0.10))",
}


def badge_style(kind: str = "idle") -> str:
    color = STATUS_COLORS.get(str(kind or "idle").lower(), TOKENS.colors.text_muted)
    bg = STATUS_BACKGROUNDS.get(str(kind or "idle").lower(), "transparent")
    return (
        f"border: 1px solid {color};"
        f"color: {color};"
        f"border-radius: {TOKENS.radius.pill}px;"
        f"padding: {TOKENS.spacing.xs}px {TOKENS.spacing.sm}px;"
        f"background: {bg};"
        f"font-size: {TOKENS.typography.caption}px;"
        "font-weight: 700;"
    )


def card_style(accent: str | None = None, elevated: bool = True) -> str:
    border = accent or TOKENS.colors.border_soft
    shadow_note = f"/* shadow: {TOKENS.shadows.md}; */" if elevated else ""
    return (
        f"background: {TOKENS.colors.surface_glass};"
        f"border: 1px solid {border};"
        f"border-radius: {TOKENS.radius.xl}px;"
        f"padding: {TOKENS.spacing.lg}px;"
        f"{shadow_note}"
    )


def card_hover_style(accent: str | None = None) -> str:
    border = accent or TOKENS.colors.primary
    return (
        f"background: {TOKENS.colors.surface_hover};"
        f"border: 1px solid {border};"
        f"border-radius: {TOKENS.radius.xl}px;"
    )


def hero_panel_style(workflow: str = "home") -> str:
    gradient = WORKFLOW_GRADIENTS.get(workflow, WORKFLOW_GRADIENTS["home"])
    return (
        f"background: {gradient};"
        f"border: 1px solid {TOKENS.colors.border_soft};"
        f"border-radius: {TOKENS.radius.xl}px;"
        f"padding: {TOKENS.spacing.xl}px;"
    )


def primary_button_style(workflow: str = "home") -> str:
    accent = WORKFLOW_ACCENTS.get(workflow, TOKENS.colors.primary)
    return (
        "QPushButton, QToolButton {"
        f"background: {accent};"
        "color: #0b1020;"
        f"border-radius: {TOKENS.radius.lg}px;"
        f"padding: {TOKENS.spacing.sm}px {TOKENS.spacing.lg}px;"
        "font-weight: 700;"
        "border: none;"
        "}"
        "QPushButton:hover, QToolButton:hover {"
        "filter: brightness(1.08);"
        "}"
    )


def secondary_button_style() -> str:
    return (
        "QPushButton, QToolButton {"
        f"background: {TOKENS.colors.surface_alt};"
        f"color: {TOKENS.colors.text};"
        f"border: 1px solid {TOKENS.colors.border_soft};"
        f"border-radius: {TOKENS.radius.lg}px;"
        f"padding: {TOKENS.spacing.sm}px {TOKENS.spacing.lg}px;"
        "}"
        "QPushButton:hover, QToolButton:hover {"
        f"background: {TOKENS.colors.surface_hover};"
        "}"
    )


def inspector_section_style() -> str:
    return (
        f"background: {TOKENS.colors.surface_alt};"
        f"border: 1px solid {TOKENS.colors.border_soft};"
        f"border-radius: {TOKENS.radius.lg}px;"
        f"padding: {TOKENS.spacing.md}px;"
    )


def app_shell_style() -> str:
    return (
        f"background: {TOKENS.colors.background};"
        f"color: {TOKENS.colors.text};"
        f"font-size: {TOKENS.typography.body}px;"
    )
