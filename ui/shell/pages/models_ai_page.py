"""Models / AI page wired to actual module registries."""

from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, QPushButton, QProgressBar, QListWidget, QListWidgetItem
from qtpy.QtCore import Qt

from modules import GET_VALID_TRANSLATORS, GET_VALID_OCR, GET_VALID_INPAINTERS, GET_VALID_TEXTDETECTORS
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

        # Query actual registries
        translators = GET_VALID_TRANSLATORS()
        ocr_modules = GET_VALID_OCR()
        inpainters = GET_VALID_INPAINTERS()
        detectors = GET_VALID_TEXTDETECTORS()

        categories = [
            ("Translators", translators, COLORS.success if translators else COLORS.warning),
            ("OCR", ocr_modules, COLORS.success if ocr_modules else COLORS.warning),
            ("Inpainters", inpainters, COLORS.success if inpainters else COLORS.warning),
            ("Text Detectors", detectors, COLORS.success if detectors else COLORS.warning),
        ]

        for i, (name, items, color) in enumerate(categories):
            card = ShellCard(name)
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{len(items)} module(s) available"))
            row.addStretch()
            row.addWidget(StatusPill("Ready" if items else "Empty", color))
            card.layout.addLayout(row)

            list_w = QListWidget()
            list_w.setMaximumHeight(120)
            for item in items[:10]:
                list_w.addItem(QListWidgetItem(str(item)))
            if len(items) > 10:
                list_w.addItem(QListWidgetItem(f"... and {len(items)-10} more"))
            card.layout.addWidget(list_w)
            grid.addWidget(card, i // 2, i % 2)

        root.addLayout(grid)

        actions = ShellCard("Model Actions")
        row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Status")
        refresh_btn.clicked.connect(self._refresh)
        row.addWidget(AccentButton("Model Manager"))
        row.addWidget(QPushButton("Download Models"))
        row.addWidget(refresh_btn)
        row.addStretch()
        actions.layout.addLayout(row)
        root.addWidget(actions)
        root.addStretch()

    def _refresh(self):
        # Re-query and update UI
        # TODO: implement dynamic refresh without rebuilding entire layout
        pass
