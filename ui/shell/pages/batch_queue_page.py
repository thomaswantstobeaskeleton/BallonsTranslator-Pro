"""Batch Queue page matching the mockup section."""

from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar, QPushButton

from ..theme import COLORS, SPACING
from .components import ShellCard, PageHeader, StatusPill, AccentButton


class BatchQueuePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        root.setSpacing(SPACING.xl)
        root.addWidget(PageHeader("Batch Queue", "Process multiple projects with pause/resume/cancel controls."))

        queue = ShellCard("Active Jobs")
        self.list = QListWidget()
        for name, status, pct in [
            ("One Piece Ch. 1088", "Running", 62),
            ("Jujutsu Kaisen Ch. 236", "Running", 41),
            ("Bleach Ch. 686", "Queued", 0),
            ("Chainsaw Man Ch. 162", "Queued", 0),
        ]:
            item = QListWidgetItem(f"{name}    {status}    {pct}%")
            self.list.addItem(item)
        queue.layout.addWidget(self.list)
        root.addWidget(queue, 1)

        controls = ShellCard("Queue Controls")
        row = QHBoxLayout()
        for text in ["Add Job", "Pause All", "Resume", "Cancel Selected", "Clear Completed"]:
            row.addWidget(AccentButton(text) if text == "Add Job" else QPushButton(text))
        row.addStretch()
        controls.layout.addLayout(row)
        root.addWidget(controls)
