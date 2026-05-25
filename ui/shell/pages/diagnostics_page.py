"""Diagnostics page matching the mockup section."""

from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QPlainTextEdit, QGridLayout

from ..theme import COLORS, SPACING
from .components import ShellCard, PageHeader, StatusPill, AccentButton


class DiagnosticsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        root.setSpacing(SPACING.xl)
        root.addWidget(PageHeader("Diagnostics", "System, rendering, OCR, model, and pipeline health checks."))

        grid = QGridLayout()
        grid.setSpacing(SPACING.lg)
        checks = [("GPU", "Good"), ("VRAM", "Good"), ("RAM", "Good"), ("Disk Space", "Good"), ("Python", "Good"), ("Dependencies", "Good")]
        for i, (name, state) in enumerate(checks):
            card = ShellCard(name)
            row = QHBoxLayout()
            row.addWidget(QLabel("Status"))
            row.addStretch()
            row.addWidget(StatusPill(state, COLORS.success))
            card.layout.addLayout(row)
            pb = QProgressBar()
            pb.setValue(100)
            card.layout.addWidget(pb)
            grid.addWidget(card, i // 3, i % 3)
        root.addLayout(grid)

        log_card = ShellCard("Diagnostics Log")
        log = QPlainTextEdit()
        log.setReadOnly(True)
        log.setPlainText("All systems operational.\nNo warnings detected.")
        log_card.layout.addWidget(log)
        row = QHBoxLayout()
        row.addWidget(AccentButton("Run Full Diagnostics"))
        row.addWidget(QPushButton("Export Report"))
        row.addStretch()
        log_card.layout.addLayout(row)
        root.addWidget(log_card, 1)
