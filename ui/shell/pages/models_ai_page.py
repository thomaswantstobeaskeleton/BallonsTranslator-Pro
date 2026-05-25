"""Models / AI page matching the mockup section."""

from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, QPushButton, QProgressBar
from qtpy.QtCore import Qt

from ..theme import COLORS, SPACING, FONTS
from .components import ShellCard, PageHeader, StatusPill, AccentButton


class ModelsAIPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        root.setSpacing(SPACING.xl)
        root.addWidget(PageHeader("Models / AI", "Manage OCR, translation, text detection, and inpainting providers."))

        grid = QGridLayout()
        grid.setSpacing(SPACING.lg)
        providers = [
            ("DeepL", "Connected", COLORS.success),
            ("Google", "Connected", COLORS.success),
            ("OpenAI", "Needs key", COLORS.warning),
            ("Anthropic", "Needs key", COLORS.warning),
            ("Local OCR", "Ready", COLORS.success),
            ("Inpaint", "Ready", COLORS.success),
        ]
        for i, (name, status, color) in enumerate(providers):
            card = ShellCard(name)
            row = QHBoxLayout()
            row.addWidget(QLabel("Provider status"))
            row.addStretch()
            row.addWidget(StatusPill(status, color))
            card.layout.addLayout(row)
            pb = QProgressBar()
            pb.setValue(100 if color == COLORS.success else 35)
            card.layout.addWidget(pb)
            grid.addWidget(card, i // 3, i % 3)
        root.addLayout(grid)

        actions = ShellCard("Model Actions")
        row = QHBoxLayout()
        for text in ["Model Manager", "Download Models", "Validate Providers", "Refresh Status"]:
            btn = AccentButton(text) if text in {"Model Manager", "Download Models"} else QPushButton(text)
            row.addWidget(btn)
        row.addStretch()
        actions.layout.addLayout(row)
        root.addWidget(actions)
        root.addStretch()
