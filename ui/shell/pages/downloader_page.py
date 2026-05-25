"""Downloader page matching the mockup section."""

from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton, QListWidget, QListWidgetItem, QProgressBar

from ..theme import SPACING, COLORS
from .components import ShellCard, PageHeader, AccentButton, StatusPill


class DownloaderPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        root.setSpacing(SPACING.xl)
        root.addWidget(PageHeader("Downloader", "Source manga/comic chapters and prepare them for translation."))

        setup = ShellCard("Download Setup")
        row = QHBoxLayout()
        source = QComboBox(); source.addItems(["MangaDex", "Comic Source", "Local URL", "Custom"])
        quality = QComboBox(); quality.addItems(["High", "Medium", "Low"])
        query = QLineEdit(); query.setPlaceholderText("Series / chapter URL / search query")
        row.addWidget(QLabel("Source")); row.addWidget(source)
        row.addWidget(QLabel("Quality")); row.addWidget(quality)
        row.addWidget(query, 1)
        row.addWidget(AccentButton("Start Download"))
        setup.layout.addLayout(row)
        root.addWidget(setup)

        jobs = ShellCard("Downloads")
        listw = QListWidget()
        for name, pct in [("One Piece Ch. 1088", 62), ("Chainsaw Man Ch. 162", 34), ("Recent thumbnails", 100)]:
            listw.addItem(QListWidgetItem(f"{name}    {pct}%"))
        jobs.layout.addWidget(listw, 1)
        bar = QProgressBar(); bar.setValue(62)
        jobs.layout.addWidget(bar)
        root.addWidget(jobs, 1)
