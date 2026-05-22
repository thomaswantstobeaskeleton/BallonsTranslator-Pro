from __future__ import annotations

from typing import Iterable, Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .design_tokens import TOKENS, badge_style, card_style, primary_button_style, secondary_button_style
from .experimental_app_shell import ExperimentalAppShell
from .job_status_drawer import JobStatusSpec


class ExperimentalShellPreviewDialog(QDialog):
    """Safe preview window for the advanced modern app shell.

    This dialog deliberately does not replace MainWindow. It lets developers and
    testers open the Dango-inspired, professional shell layout while keeping the
    production editor untouched.
    """

    dashboard_action_requested = Signal(str, str)
    workflow_requested = Signal(str)
    mode_changed = Signal(str)
    translation_assist_requested = Signal()
    ocr_rerun_requested = Signal()
    layout_review_requested = Signal()
    typography_qa_requested = Signal()
    job_cancel_requested = Signal(str)
    job_pause_requested = Signal(str)
    job_details_requested = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ExperimentalShellPreviewDialog")
        self.setWindowTitle(self.tr("Experimental Advanced UI Shell Preview"))
        self.resize(1320, 860)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(TOKENS.spacing.lg, TOKENS.spacing.lg, TOKENS.spacing.lg, TOKENS.spacing.lg)
        root.setSpacing(TOKENS.spacing.md)

        header = QVBoxLayout()
        title_row = QHBoxLayout()
        title = QLabel(self.tr("Advanced UI shell preview"), self)
        title.setStyleSheet(f"font-size: {TOKENS.typography.headline}px; font-weight: 900; color: {TOKENS.colors.text};")
        title_row.addWidget(title, 1)
        badge = QLabel(self.tr("Experimental"), self)
        badge.setStyleSheet(badge_style("experimental"))
        title_row.addWidget(badge)
        header.addLayout(title_row)

        subtitle = QLabel(
            self.tr(
                "Preview the modern, dense, professional shell without replacing the existing editor. "
                "Actions route through safe dispatcher hooks when wired."
            ),
            self,
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"font-size: {TOKENS.typography.body}px; color: {TOKENS.colors.text_muted};")
        header.addWidget(subtitle)
        root.addLayout(header)

        self.shell = ExperimentalAppShell(self)
        self.shell.workflow_requested.connect(self.workflow_requested.emit)
        self.shell.dashboard_action_requested.connect(self.dashboard_action_requested.emit)
        self.shell.mode_changed.connect(self.mode_changed.emit)
        self.shell.translation_assist_requested.connect(self.translation_assist_requested.emit)
        self.shell.ocr_rerun_requested.connect(self.ocr_rerun_requested.emit)
        self.shell.layout_review_requested.connect(self.layout_review_requested.emit)
        self.shell.typography_qa_requested.connect(self.typography_qa_requested.emit)
        self.shell.job_cancel_requested.connect(self.job_cancel_requested.emit)
        self.shell.job_pause_requested.connect(self.job_pause_requested.emit)
        self.shell.job_details_requested.connect(self.job_details_requested.emit)
        root.addWidget(self.shell, 1)

        footer = QHBoxLayout()
        warning = QLabel(self.tr("This is a preview surface. Legacy menus and the existing editor remain the source of truth."), self)
        warning.setWordWrap(True)
        warning.setStyleSheet(card_style(TOKENS.colors.warning, elevated=False))
        footer.addWidget(warning, 1)

        self.sample_jobs_btn = QPushButton(self.tr("Add sample jobs"), self)
        self.sample_jobs_btn.setStyleSheet(secondary_button_style())
        self.sample_jobs_btn.clicked.connect(self.add_sample_jobs)
        footer.addWidget(self.sample_jobs_btn)

        close_btn = QPushButton(self.tr("Close"), self)
        close_btn.setStyleSheet(primary_button_style("home"))
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        root.addLayout(footer)

    def set_current_mode(self, mode: str):
        self.shell.set_current_mode(mode)

    def refresh_jobs(self, jobs: Iterable[JobStatusSpec]):
        self.shell.refresh_jobs(jobs)

    def add_sample_jobs(self):
        self.refresh_jobs(
            [
                JobStatusSpec(
                    job_id="sample-ocr",
                    title=self.tr("OCR current page"),
                    kind="OCR",
                    status="running",
                    progress=42,
                    detail=self.tr("Running selected OCR provider on detected text blocks."),
                    can_cancel=True,
                ),
                JobStatusSpec(
                    job_id="sample-export",
                    title=self.tr("Export proof pack"),
                    kind="Export",
                    status="warning",
                    progress=78,
                    detail=self.tr("Preparing reviewer assets and manifest."),
                    can_cancel=True,
                    warnings=[self.tr("2 blocks may overflow their bubbles"), self.tr("1 missing font fallback")],
                ),
            ]
        )


def open_experimental_shell_preview(parent: Optional[QWidget] = None) -> ExperimentalShellPreviewDialog:
    dialog = ExperimentalShellPreviewDialog(parent)
    dialog.show()
    return dialog
