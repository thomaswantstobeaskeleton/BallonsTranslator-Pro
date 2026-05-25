"""Job Status Drawer foundation (mockup section 12)."""

from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar, QPushButton
from qtpy.QtCore import Qt

from .theme import COLORS, FONTS, SPACING, RADIUS


class JobStatusDrawer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("JobStatusDrawer")
        self.setFixedHeight(180)
        self.setStyleSheet(f"""
            #JobStatusDrawer {{
                background-color: {COLORS.bg_elevated};
                border-top: 1px solid {COLORS.border};
            }}
        """)
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.lg, SPACING.md, SPACING.lg, SPACING.md)
        root.setSpacing(SPACING.sm)

        header = QHBoxLayout()
        title = QLabel("Job Status")
        title.setStyleSheet(f"color: {COLORS.text_primary}; font-size: {FONTS.size_lg}px; font-weight: 800; background: transparent;")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(QPushButton("Pause All"))
        header.addWidget(QPushButton("Clear Completed"))
        root.addLayout(header)

        row = QHBoxLayout()
        self.jobs = QListWidget()
        for item in ["page_001.png    OCR    Running", "page_002.png    Translate    Queued", "Export    Waiting"]:
            self.jobs.addItem(QListWidgetItem(item))
        row.addWidget(self.jobs, 2)

        details = QVBoxLayout()
        details.addWidget(QLabel("Overall Progress"))
        self.overall = QProgressBar(); self.overall.setValue(55)
        details.addWidget(self.overall)
        details.addWidget(QLabel("Latest: translated 14 blocks"))
        details.addStretch()
        row.addLayout(details, 1)
        root.addLayout(row, 1)
