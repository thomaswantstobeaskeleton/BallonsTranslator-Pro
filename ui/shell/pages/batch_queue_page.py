"""Batch Queue page with functional job list and controls."""

from __future__ import annotations
import os.path as osp

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QProgressBar, QPushButton, QFileDialog, QMessageBox,
)
from qtpy.QtCore import Qt

from ..theme import COLORS, SPACING
from .components import ShellCard, PageHeader, StatusPill, AccentButton


class BatchQueuePage(QWidget):
    """Functional batch queue for processing multiple project folders."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs: list[dict] = []  # {path, name, status, pct}
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        root.setSpacing(SPACING.xl)
        root.addWidget(PageHeader("Batch Queue", "Process multiple projects with pause/resume/cancel controls."))

        queue = ShellCard("Active Jobs")
        self._list = QListWidget()
        queue.layout.addWidget(self._list, 1)
        root.addWidget(queue, 1)

        controls = ShellCard("Queue Controls")
        row = QHBoxLayout()
        add_btn = AccentButton("Add Job")
        add_btn.clicked.connect(self._add_job)
        row.addWidget(add_btn)

        pause_btn = QPushButton("Pause All")
        pause_btn.clicked.connect(self._pause_all)
        row.addWidget(pause_btn)

        resume_btn = QPushButton("Resume")
        resume_btn.clicked.connect(self._resume_all)
        row.addWidget(resume_btn)

        cancel_btn = QPushButton("Cancel Selected")
        cancel_btn.clicked.connect(self._cancel_selected)
        row.addWidget(cancel_btn)

        clear_btn = QPushButton("Clear Completed")
        clear_btn.clicked.connect(self._clear_completed)
        row.addWidget(clear_btn)

        row.addStretch()
        controls.layout.addLayout(row)
        root.addWidget(controls)

    def _add_job(self):
        path = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if path:
            name = osp.basename(path)
            self._jobs.append({"path": path, "name": name, "status": "Queued", "pct": 0})
            self._refresh_list()

    def _pause_all(self):
        for job in self._jobs:
            if job["status"] == "Running":
                job["status"] = "Paused"
        self._refresh_list()

    def _resume_all(self):
        for job in self._jobs:
            if job["status"] in ("Paused", "Queued"):
                job["status"] = "Running"
        self._refresh_list()

    def _cancel_selected(self):
        row = self._list.currentRow()
        if 0 <= row < len(self._jobs):
            self._jobs[row]["status"] = "Cancelled"
            self._refresh_list()

    def _clear_completed(self):
        self._jobs = [j for j in self._jobs if j["status"] not in ("Completed", "Cancelled")]
        self._refresh_list()

    def _refresh_list(self):
        self._list.clear()
        for job in self._jobs:
            item = QListWidgetItem(f"{job['name']}    {job['status']}    {job['pct']}%")
            self._list.addItem(item)
