from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .design_tokens import TOKENS, badge_style, card_style


@dataclass(frozen=True)
class JobStatusSpec:
    job_id: str
    title: str
    kind: str = "pipeline"
    status: str = "idle"
    progress: int = 0
    detail: str = ""
    can_cancel: bool = False
    can_pause: bool = False
    warnings: List[str] = field(default_factory=list)


class JobStatusRow(QFrame):
    cancel_requested = Signal(str)
    pause_requested = Signal(str)
    details_requested = Signal(str)

    def __init__(self, spec: JobStatusSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("JobStatusRow")
        self.setStyleSheet(card_style())
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(TOKENS.spacing.md, TOKENS.spacing.md, TOKENS.spacing.md, TOKENS.spacing.md)
        layout.setSpacing(TOKENS.spacing.sm)

        header = QHBoxLayout()
        title = QLabel(self.spec.title)
        title.setObjectName("JobStatusTitle")
        title.setStyleSheet(f"font-size: {TOKENS.typography.body_large}px; font-weight: 700;")
        header.addWidget(title, 1)

        kind = QLabel(self.spec.kind)
        kind.setStyleSheet(badge_style("idle"))
        header.addWidget(kind)

        status = QLabel(self.spec.status)
        status.setStyleSheet(badge_style(self.spec.status))
        header.addWidget(status)
        layout.addLayout(header)

        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(max(0, min(100, int(self.spec.progress))))
        layout.addWidget(self.progress)

        if self.spec.detail:
            detail = QLabel(self.spec.detail)
            detail.setWordWrap(True)
            detail.setStyleSheet(f"font-size: {TOKENS.typography.caption}px; color: {TOKENS.colors.text_muted};")
            layout.addWidget(detail)

        if self.spec.warnings:
            warning = QLabel("; ".join(str(w) for w in self.spec.warnings[:3]))
            warning.setWordWrap(True)
            warning.setStyleSheet(f"font-size: {TOKENS.typography.caption}px; color: {TOKENS.colors.warning};")
            layout.addWidget(warning)

        actions = QHBoxLayout()
        actions.addStretch(1)
        details_btn = QPushButton(self.tr("Details"), self)
        details_btn.clicked.connect(lambda: self.details_requested.emit(self.spec.job_id))
        actions.addWidget(details_btn)
        if self.spec.can_pause:
            pause_btn = QPushButton(self.tr("Pause"), self)
            pause_btn.clicked.connect(lambda: self.pause_requested.emit(self.spec.job_id))
            actions.addWidget(pause_btn)
        if self.spec.can_cancel:
            cancel_btn = QPushButton(self.tr("Cancel"), self)
            cancel_btn.clicked.connect(lambda: self.cancel_requested.emit(self.spec.job_id))
            actions.addWidget(cancel_btn)
        layout.addLayout(actions)


class JobStatusDrawer(QWidget):
    """Bottom job/status drawer for long-running workflows.

    This is a shell component for OCR, translation, inpaint, render, export,
    raw download, live translation, model download, batch queue, and Translation
    Assist provider jobs.  It is standalone so it can be introduced beside the
    legacy progress UI before becoming the final bottom drawer.
    """

    cancel_requested = Signal(str)
    pause_requested = Signal(str)
    details_requested = Signal(str)
    clear_completed_requested = Signal()

    def __init__(self, jobs: Optional[Iterable[JobStatusSpec]] = None, parent=None):
        super().__init__(parent)
        self._jobs: Dict[str, JobStatusSpec] = {j.job_id: j for j in (jobs or [])}
        self._rows: Dict[str, JobStatusRow] = {}
        self._expanded = True
        self._build_ui()
        self.refresh_jobs(self._jobs.values())

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(TOKENS.spacing.sm, TOKENS.spacing.sm, TOKENS.spacing.sm, TOKENS.spacing.sm)
        root.setSpacing(TOKENS.spacing.sm)

        header = QHBoxLayout()
        self.toggle_btn = QToolButton(self)
        self.toggle_btn.setText("▾")
        self.toggle_btn.setToolTip(self.tr("Show or hide job details"))
        self.toggle_btn.clicked.connect(self.toggle_expanded)
        header.addWidget(self.toggle_btn)

        title = QLabel(self.tr("Jobs & Status"), self)
        title.setObjectName("JobStatusDrawerTitle")
        title.setStyleSheet(f"font-size: {TOKENS.typography.title}px; font-weight: 700;")
        header.addWidget(title)

        self.summary = QLabel(self.tr("No active jobs"), self)
        self.summary.setStyleSheet(f"color: {TOKENS.colors.text_muted};")
        header.addWidget(self.summary, 1)

        clear_btn = QPushButton(self.tr("Clear completed"), self)
        clear_btn.clicked.connect(self.clear_completed_requested.emit)
        header.addWidget(clear_btn)
        root.addLayout(header)

        self.list_widget = QListWidget(self)
        self.list_widget.setObjectName("JobStatusList")
        root.addWidget(self.list_widget)

    def refresh_jobs(self, jobs: Iterable[JobStatusSpec]):
        self._jobs = {j.job_id: j for j in jobs}
        self.list_widget.clear()
        self._rows.clear()
        for job in self._jobs.values():
            item = QListWidgetItem(self.list_widget)
            row = JobStatusRow(job, self.list_widget)
            row.cancel_requested.connect(self.cancel_requested.emit)
            row.pause_requested.connect(self.pause_requested.emit)
            row.details_requested.connect(self.details_requested.emit)
            item.setSizeHint(row.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, row)
            self._rows[job.job_id] = row
        self._update_summary()

    def upsert_job(self, job: JobStatusSpec):
        jobs = dict(self._jobs)
        jobs[job.job_id] = job
        self.refresh_jobs(jobs.values())

    def remove_job(self, job_id: str):
        jobs = dict(self._jobs)
        jobs.pop(job_id, None)
        self.refresh_jobs(jobs.values())

    def toggle_expanded(self):
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool):
        self._expanded = bool(expanded)
        self.list_widget.setVisible(self._expanded)
        self.toggle_btn.setText("▾" if self._expanded else "▸")

    def job_ids(self) -> List[str]:
        return list(self._jobs.keys())

    def _update_summary(self):
        total = len(self._jobs)
        if total == 0:
            self.summary.setText(self.tr("No active jobs"))
            return
        running = sum(1 for job in self._jobs.values() if str(job.status).lower() in {"running", "capturing", "ocr", "translating", "exporting"})
        warnings = sum(1 for job in self._jobs.values() if job.warnings or str(job.status).lower() in {"warning", "error"})
        self.summary.setText(self.tr("{running} running · {total} total · {warnings} need attention").format(running=running, total=total, warnings=warnings))
