"""Compact project and status footer matching the mockup bottom strip."""

from __future__ import annotations
from qtpy.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar, QPushButton
from qtpy.QtCore import Qt

from .theme import COLORS, FONTS, SPACING


class StatusFooter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusFooter")
        self.setFixedHeight(28)
        self.setStyleSheet(f"""
            #StatusFooter {{
                background-color: {COLORS.bg_deepest};
                border-top: 1px solid {COLORS.border_subtle};
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING.lg, 2, SPACING.lg, 2)
        layout.setSpacing(SPACING.lg)

        self.project_label = QLabel("No project open")
        self.project_label.setStyleSheet(f"color: {COLORS.text_muted}; font-size: {FONTS.size_xs}px; background: transparent;")
        layout.addWidget(self.project_label)

        layout.addStretch()

        self.page_label = QLabel("Page: - / -")
        self.page_label.setStyleSheet(f"color: {COLORS.text_muted}; font-size: {FONTS.size_xs}px; background: transparent;")
        layout.addWidget(self.page_label)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(120)
        self.progress.setFixedHeight(6)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {COLORS.text_muted}; font-size: {FONTS.size_xs}px; background: transparent;")
        layout.addWidget(self.status_label)

        self.version_label = QLabel("v2.0.0")
        self.version_label.setStyleSheet(f"color: {COLORS.text_muted}; font-size: {FONTS.size_xs}px; background: transparent;")
        layout.addWidget(self.version_label)
