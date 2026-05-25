"""Reusable QWidget components for shell pages."""

from __future__ import annotations
from typing import Optional

from qtpy.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget
from qtpy.QtCore import Qt

from ..theme import COLORS, FONTS, SPACING, RADIUS


class ShellCard(QFrame):
    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ShellCard")
        self.setStyleSheet(f"""
            #ShellCard {{
                background-color: {COLORS.bg_surface};
                border: 1px solid {COLORS.border};
                border-radius: {RADIUS.lg}px;
            }}
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(SPACING.lg, SPACING.lg, SPACING.lg, SPACING.lg)
        self.layout.setSpacing(SPACING.md)
        if title:
            lbl = QLabel(title)
            lbl.setStyleSheet(f"color: {COLORS.text_primary}; font-size: {FONTS.size_lg}px; font-weight: 700; background: transparent;")
            self.layout.addWidget(lbl)


class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {COLORS.text_primary}; font-size: {FONTS.size_h2}px; font-weight: 800; background: transparent;")
        lay.addWidget(title_lbl)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setStyleSheet(f"color: {COLORS.text_secondary}; font-size: {FONTS.size_base}px; background: transparent;")
            lay.addWidget(sub)


class StatusPill(QLabel):
    def __init__(self, text: str, color: str = COLORS.success, parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(22)
        self.setStyleSheet(f"""
            background-color: {color};
            color: {COLORS.text_inverse};
            border-radius: 11px;
            padding: 2px 10px;
            font-size: {FONTS.size_xs}px;
            font-weight: 700;
        """)


class AccentButton(QPushButton):
    def __init__(self, text: str, parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setProperty("accent", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
