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


@dataclass(frozen=True)
class RadiusTokens:
    sm: int = 4
    md: int = 8
    lg: int = 12
    xl: int = 16


@dataclass(frozen=True)
class TypographyTokens:
    caption: int = 11
    body: int = 13
    body_large: int = 14
    title: int = 16
    headline: int = 20


@dataclass(frozen=True)
class ColorTokens:
    background: str = "#111318"
    surface: str = "#1a1d24"
    surface_alt: str = "#222733"
    border: str = "#303645"
    text: str = "#f2f4f8"
    text_muted: str = "#aab2c0"
    primary: str = "#7aa2ff"
    success: str = "#5cc887"
    warning: str = "#f4c76b"
    danger: str = "#ff7b7b"
    info: str = "#7fd7ff"


@dataclass(frozen=True)
class UITokens:
    spacing: SpacingTokens = SpacingTokens()
    radius: RadiusTokens = RadiusTokens()
    typography: TypographyTokens = TypographyTokens()
    colors: ColorTokens = ColorTokens()


TOKENS = UITokens()

STATUS_COLORS: Dict[str, str] = {
    "idle": TOKENS.colors.text_muted,
    "running": TOKENS.colors.info,
    "success": TOKENS.colors.success,
    "warning": TOKENS.colors.warning,
    "error": TOKENS.colors.danger,
    "disabled": TOKENS.colors.text_muted,
    "experimental": TOKENS.colors.warning,
}

WORKFLOW_ACCENTS: Dict[str, str] = {
    "home": TOKENS.colors.primary,
    "editor": TOKENS.colors.primary,
    "live": TOKENS.colors.info,
    "quick_image": TOKENS.colors.success,
    "downloader": TOKENS.colors.warning,
    "batch": TOKENS.colors.info,
    "assist": TOKENS.colors.success,
    "models": TOKENS.colors.primary,
    "settings": TOKENS.colors.text_muted,
    "diagnostics": TOKENS.colors.warning,
}


def badge_style(kind: str = "idle") -> str:
    color = STATUS_COLORS.get(str(kind or "idle"), TOKENS.colors.text_muted)
    return (
        f"border: 1px solid {color};"
        f"color: {color};"
        f"border-radius: {TOKENS.radius.md}px;"
        f"padding: {TOKENS.spacing.xs}px {TOKENS.spacing.sm}px;"
        "background: transparent;"
    )


def card_style(accent: str | None = None) -> str:
    border = accent or TOKENS.colors.border
    return (
        f"background: {TOKENS.colors.surface};"
        f"border: 1px solid {border};"
        f"border-radius: {TOKENS.radius.lg}px;"
        f"padding: {TOKENS.spacing.lg}px;"
    )


def inspector_section_style() -> str:
    return (
        f"background: {TOKENS.colors.surface_alt};"
        f"border: 1px solid {TOKENS.colors.border};"
        f"border-radius: {TOKENS.radius.md}px;"
        f"padding: {TOKENS.spacing.md}px;"
    )
