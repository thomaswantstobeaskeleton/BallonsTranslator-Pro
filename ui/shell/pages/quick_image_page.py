"""Quick Image page matching the mockup section."""

from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QFrame
from qtpy.QtCore import Qt

from ..theme import COLORS, SPACING, FONTS, RADIUS
from .components import ShellCard, PageHeader, AccentButton


class QuickImagePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        root.setSpacing(SPACING.xl)
        root.addWidget(PageHeader("Quick Image", "Drag-and-drop one image for a fast OCR + translate pass."))

        row = QHBoxLayout()
        drop = ShellCard("Drop Image")
        drop_zone = QLabel("Drag & drop image here\nor click Open Image")
        drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_zone.setMinimumHeight(260)
        drop_zone.setStyleSheet(f"""
            background-color: {COLORS.bg_base};
            border: 1px dashed {COLORS.border_focus};
            border-radius: {RADIUS.lg}px;
            color: {COLORS.text_secondary};
            font-size: {FONTS.size_lg}px;
            font-weight: 600;
        """)
        drop.layout.addWidget(drop_zone, 1)
        drop.layout.addWidget(AccentButton("Open Image"))
        row.addWidget(drop, 2)

        recent = ShellCard("Recent Images")
        listw = QListWidget()
        for item in ["page_001.png", "panel_crop.png", "cover_scan.jpg", "quick_test.webp"]:
            listw.addItem(QListWidgetItem(item))
        recent.layout.addWidget(listw)
        row.addWidget(recent, 1)
        root.addLayout(row, 1)

        actions = ShellCard("Quick Pipeline")
        btn_row = QHBoxLayout()
        for text in ["OCR Only", "Translate", "Inpaint + Letter", "Export"]:
            btn_row.addWidget(AccentButton(text) if text == "Translate" else QPushButton(text))
        btn_row.addStretch()
        actions.layout.addLayout(btn_row)
        root.addWidget(actions)
