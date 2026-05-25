"""Settings page shell matching the mockup section."""

from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSpinBox, QPushButton, QGridLayout

from ..theme import COLORS, SPACING
from .components import ShellCard, PageHeader, AccentButton


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        root.setSpacing(SPACING.xl)
        root.addWidget(PageHeader("Settings", "General, OCR, translator, export, UI/theme, and advanced options."))

        grid = QGridLayout()
        grid.setSpacing(SPACING.lg)
        for idx, title in enumerate(["General", "OCR", "Translator", "Export", "UI / Theme", "Advanced"]):
            card = ShellCard(title)
            cb = QComboBox()
            cb.addItems(["Default", "Balanced", "Performance", "Custom"])
            card.layout.addWidget(QLabel("Profile"))
            card.layout.addWidget(cb)
            card.layout.addWidget(QCheckBox("Enable recommended defaults"))
            grid.addWidget(card, idx // 3, idx % 3)
        root.addLayout(grid)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(QPushButton("Reset"))
        row.addWidget(AccentButton("Save Settings"))
        root.addLayout(row)
        root.addStretch()
