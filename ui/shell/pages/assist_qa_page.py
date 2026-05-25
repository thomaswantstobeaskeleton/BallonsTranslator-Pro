"""Assist / QA page matching the mockup section."""

from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPlainTextEdit, QListWidget, QListWidgetItem, QPushButton, QSplitter
from qtpy.QtCore import Qt

from ..theme import COLORS, SPACING
from .components import ShellCard, PageHeader, AccentButton


class AssistQAPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        root.setSpacing(SPACING.xl)
        root.addWidget(PageHeader("Assist / QA", "Compare translation candidates, glossary hints, and quality warnings."))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        assistant = ShellCard("AI Assistant")
        prompt = QPlainTextEdit()
        prompt.setPlaceholderText("Ask for translation improvements, style checks, glossary suggestions...")
        assistant.layout.addWidget(prompt, 1)
        row = QHBoxLayout()
        row.addWidget(AccentButton("Improve this Translation"))
        row.addWidget(QPushButton("Run QA"))
        row.addStretch()
        assistant.layout.addLayout(row)
        splitter.addWidget(assistant)

        candidates = ShellCard("Candidates / Warnings")
        listw = QListWidget()
        for text in [
            "DeepL: Natural phrasing candidate",
            "OpenAI: More contextual candidate",
            "Glossary: 2 matched terms",
            "QA: No critical issues",
        ]:
            listw.addItem(QListWidgetItem(text))
        candidates.layout.addWidget(listw)
        splitter.addWidget(candidates)
        splitter.setSizes([620, 360])
        root.addWidget(splitter, 1)
