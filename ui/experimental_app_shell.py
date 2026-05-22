from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from qtpy.QtCore import QByteArray, Qt, Signal
from qtpy.QtWidgets import QSplitter, QStackedWidget, QVBoxLayout, QWidget

from .design_tokens import TOKENS, app_shell_style
from .editor_inspector import EditorInspector
from .job_status_drawer import JobStatusDrawer, JobStatusSpec
from .mode_dashboard import ModeDashboard, dashboard_for_mode
from .mode_rail import DEFAULT_MODE_RAIL_ITEMS, ModeRail, ModeRailItemSpec
from .workflow_home import WorkflowHomeWidget


@dataclass(frozen=True)
class ShellPageSpec:
    key: str
    title: str
    description: str


DEFAULT_SHELL_PAGES: tuple[ShellPageSpec, ...] = (
    ShellPageSpec("home", "Home", "Choose a workflow or continue recent work."),
    ShellPageSpec("editor", "Editor", "Manga/comic localization workspace."),
    ShellPageSpec("live", "Live Translation", "Realtime screen or Chrome manhua translation."),
    ShellPageSpec("quick_image", "Image Quick Translation", "Drop images, translate quickly, and optionally promote to a project."),
    ShellPageSpec("downloader", "Raw Downloader", "Search sources, choose chapters, and import raws."),
    ShellPageSpec("batch", "Batch Queue", "Process many chapters, folders, or archives."),
    ShellPageSpec("assist", "Translation Assist / QA", "Provider comparison, TM, glossary, concordance, SFX, and QA."),
    ShellPageSpec("models", "Models & Providers", "Install models, test providers, and inspect runtime health."),
    ShellPageSpec("settings", "Settings", "Searchable app and workflow settings."),
    ShellPageSpec("diagnostics", "Diagnostics / Help", "Logs, doctors, reports, documentation, and support."),
)


class ExperimentalAppShell(QWidget):
    """Composable app-shell prototype for the phased UI/UX rework.

    Modern here means advanced and polished: dashboards expose professional
    actions, metrics, warnings, and notes instead of hiding complexity.
    """

    mode_changed = Signal(str)
    workflow_requested = Signal(str)
    dashboard_action_requested = Signal(str, str)
    translation_assist_requested = Signal()
    ocr_rerun_requested = Signal()
    layout_review_requested = Signal()
    typography_qa_requested = Signal()
    job_cancel_requested = Signal(str)
    job_pause_requested = Signal(str)
    job_details_requested = Signal(str)

    def __init__(
        self,
        pages: Optional[Iterable[ShellPageSpec]] = None,
        rail_items: Optional[Iterable[ModeRailItemSpec]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("ExperimentalAppShell")
        self.setStyleSheet(
            "QWidget#ExperimentalAppShell {" + app_shell_style() + "}"
            "QSplitter::handle { background: rgba(255, 255, 255, 0.05); }"
            "QSplitter::handle:hover { background: rgba(122, 162, 255, 0.28); }"
            "QStackedWidget#ExperimentalAppShellCenterStack {"
            f"background: {TOKENS.colors.background_soft};"
            f"border-left: 1px solid {TOKENS.colors.border_soft};"
            f"border-right: 1px solid {TOKENS.colors.border_soft};"
            "}"
        )
        self.pages: List[ShellPageSpec] = list(pages or DEFAULT_SHELL_PAGES)
        self._page_indexes: Dict[str, int] = {}
        self._build_ui(rail_items)
        self.set_current_mode("home")

    def _build_ui(self, rail_items: Optional[Iterable[ModeRailItemSpec]]):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.main_splitter.setObjectName("ExperimentalAppShellMainSplitter")
        self.main_splitter.setChildrenCollapsible(False)
        root.addWidget(self.main_splitter, 1)

        self.mode_rail = ModeRail(rail_items or DEFAULT_MODE_RAIL_ITEMS, self.main_splitter)
        self.mode_rail.mode_requested.connect(self.set_current_mode)
        self.main_splitter.addWidget(self.mode_rail)

        self.center_stack = QStackedWidget(self.main_splitter)
        self.center_stack.setObjectName("ExperimentalAppShellCenterStack")
        self._populate_pages()
        self.main_splitter.addWidget(self.center_stack)

        self.editor_inspector = EditorInspector(parent=self.main_splitter)
        self.editor_inspector.request_translation_assist.connect(self.translation_assist_requested.emit)
        self.editor_inspector.request_ocr_rerun.connect(self.ocr_rerun_requested.emit)
        self.editor_inspector.request_layout_review.connect(self.layout_review_requested.emit)
        self.editor_inspector.request_typography_qa.connect(self.typography_qa_requested.emit)
        self.main_splitter.addWidget(self.editor_inspector)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([128, 900, 340])

        self.job_drawer = JobStatusDrawer(parent=self)
        self.job_drawer.cancel_requested.connect(self.job_cancel_requested.emit)
        self.job_drawer.pause_requested.connect(self.job_pause_requested.emit)
        self.job_drawer.details_requested.connect(self.job_details_requested.emit)
        root.addWidget(self.job_drawer, 0)

    def _populate_pages(self):
        for spec in self.pages:
            if spec.key == "home":
                page = WorkflowHomeWidget(parent=self.center_stack)
                page.workflow_requested.connect(self.workflow_requested.emit)
                page.workflow_requested.connect(self.set_current_mode)
            else:
                dashboard_spec = dashboard_for_mode(spec.key, spec.title, spec.description)
                page = ModeDashboard(dashboard_spec, self.center_stack)
                page.action_requested.connect(self.dashboard_action_requested.emit)
            self._page_indexes[spec.key] = self.center_stack.addWidget(page)

    def set_current_mode(self, key: str):
        key = str(key or "home")
        index = self._page_indexes.get(key)
        if index is None:
            index = self._page_indexes.get("home", 0)
            key = "home"
        self.center_stack.setCurrentIndex(index)
        self.mode_rail.set_current_mode(key)
        self.editor_inspector.setVisible(key in {"editor", "assist"})
        self.mode_changed.emit(key)

    def current_mode(self) -> str:
        for key, index in self._page_indexes.items():
            if index == self.center_stack.currentIndex():
                return key
        return ""

    def page_keys(self) -> List[str]:
        return [p.key for p in self.pages]

    def save_splitter_state(self) -> bytes:
        state = self.main_splitter.saveState()
        return bytes(state)

    def restore_splitter_state(self, data: bytes | QByteArray | None) -> bool:
        if not data:
            return False
        if isinstance(data, QByteArray):
            state = data
        else:
            state = QByteArray(data)
        return bool(self.main_splitter.restoreState(state))

    def refresh_jobs(self, jobs: Iterable[JobStatusSpec]):
        self.job_drawer.refresh_jobs(jobs)
