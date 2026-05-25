"""Live Translate page matching the mockup section."""

from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QPlainTextEdit, QListWidget, QListWidgetItem, QSplitter
from qtpy.QtCore import Qt

from ..theme import COLORS, SPACING
from .components import ShellCard, PageHeader, AccentButton, StatusPill


class LiveTranslatePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        root.setSpacing(SPACING.xl)
        root.addWidget(PageHeader("Live Translate", "Screen-region OCR and translation controls. Developer realtime translator remains hidden unless dev mode is enabled."))

        top = QHBoxLayout()
        for label, values in [
            ("Source", ["Japanese", "Chinese", "Korean", "Auto Detect"]),
            ("Target", ["English", "French", "German", "Spanish"]),
            ("Provider", ["DeepL", "Google", "OpenAI", "Local"]),
        ]:
            card = ShellCard(label)
            cb = QComboBox()
            cb.addItems(values)
            card.layout.addWidget(cb)
            top.addWidget(card)
        root.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        input_card = ShellCard("Input")
        input_box = QPlainTextEdit()
        input_box.setPlaceholderText("Detected source text appears here...")
        input_card.layout.addWidget(input_box)
        input_card.layout.addWidget(AccentButton("Translate"))
        splitter.addWidget(input_card)

        history_card = ShellCard("History")
        history = QListWidget()
        for item in ["こんにちは → Hello", "ありがとうございます → Thank you", "すみません → Excuse me"]:
            history.addItem(QListWidgetItem(item))
        history_card.layout.addWidget(history)
        splitter.addWidget(history_card)
        splitter.setSizes([520, 420])
        root.addWidget(splitter, 1)
