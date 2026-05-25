"""
BallonsShell – the new top-level QMainWindow for BallonsTranslator-Pro.

Layout:
  ┌──────────┬─────────────────────────────────────────────┐
  │          │  Title bar (project info, GPU, RAM)          │
  │ Sidebar  ├─────────────────────────────────────────────┤
  │ (nav)    │  QStackedWidget (pages)                      │
  │          │                                              │
  └──────────┴─────────────────────────────────────────────┘

Each section is a QWidget page inside the stacked widget.
The sidebar drives navigation via NavController.
"""

from __future__ import annotations
import os.path as osp
import sys
from typing import Optional

from qtpy.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget,
    QLabel, QFrame, QApplication, QSizePolicy, QFileDialog,
)
from qtpy.QtCore import Qt, Signal, QSize, QTimer, QUrl
from qtpy.QtGui import QIcon, QFont, QCloseEvent, QKeySequence, QShortcut
from qtpy.QtQuickWidgets import QQuickWidget

import utils.shared as shared
from utils.config import ProgramConfig, pcfg

from .theme import COLORS, FONTS, SPACING, TITLEBAR_HEIGHT, build_shell_stylesheet
from .nav_controller import NavController, SECTIONS
from .sidebar_widget import SidebarWidget
from .top_action_bar import TopActionBar
from .status_footer import StatusFooter
from .command_palette import CommandPalette
from .job_status_drawer import JobStatusDrawer
from .pages.home_page import HomePage
from .pages.editor_page import EditorPage
from .pages.assist_qa_page import AssistQAPage
from .pages.batch_queue_page import BatchQueuePage
from .pages.diagnostics_page import DiagnosticsPage
from .pages.downloader_page import DownloaderPage
from .pages.live_translate_page import LiveTranslatePage
from .pages.models_ai_page import ModelsAIPage
from .pages.quick_image_page import QuickImagePage
from .pages.settings_page import SettingsPage
from .pages.stub_page import StubPage


class _TitleBar(QWidget):
    """Slim top bar showing project info, GPU stats, and version."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(TITLEBAR_HEIGHT)
        self.setObjectName("ShellTitleBar")
        self.setStyleSheet(f"""
            #ShellTitleBar {{
                background-color: {COLORS.bg_deepest};
                border-bottom: 1px solid {COLORS.border_subtle};
            }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(SPACING.lg, 0, SPACING.lg, 0)
        lay.setSpacing(SPACING.lg)

        self._app_label = QLabel("BallonsTranslator Pro")
        self._app_label.setStyleSheet(f"""
            color: {COLORS.accent};
            font-size: {FONTS.size_base}px;
            font-weight: 700;
            background: transparent;
        """)
        lay.addWidget(self._app_label)

        self._project_label = QLabel("")
        self._project_label.setStyleSheet(f"color: {COLORS.text_secondary}; font-size: {FONTS.size_sm}px; background: transparent;")
        lay.addWidget(self._project_label)

        lay.addStretch()

        self._gpu_label = QLabel("")
        self._gpu_label.setStyleSheet(f"color: {COLORS.text_muted}; font-size: {FONTS.size_xs}px; background: transparent;")
        lay.addWidget(self._gpu_label)

        self._ram_label = QLabel("")
        self._ram_label.setStyleSheet(f"color: {COLORS.text_muted}; font-size: {FONTS.size_xs}px; background: transparent;")
        lay.addWidget(self._ram_label)

        self._version_label = QLabel(f"v{shared.VERSION if hasattr(shared, 'VERSION') else '?'}")
        self._version_label.setStyleSheet(f"color: {COLORS.text_muted}; font-size: {FONTS.size_xs}px; background: transparent;")
        lay.addWidget(self._version_label)

    def set_project(self, name: str):
        self._project_label.setText(f"Project: {name}" if name else "")

    def set_gpu_info(self, text: str):
        self._gpu_label.setText(text)

    def set_ram_info(self, text: str):
        self._ram_label.setText(text)


class BallonsShell(QMainWindow):
    """New top-level window for BallonsTranslator-Pro."""

    restart_signal = Signal()

    def __init__(
        self,
        app: QApplication,
        config: ProgramConfig,
        open_dir: str = "",
        **kwargs,
    ):
        super().__init__()
        self._app = app
        self._config = config
        self._open_dir = open_dir
        self._kwargs = kwargs

        self.setWindowTitle("BallonsTranslator Pro")
        self.setMinimumSize(1024, 700)
        self.resize(1440, 900)

        # Apply shell theme
        self.setStyleSheet(build_shell_stylesheet())

        # ── Navigation controller ─────────────────────────────
        self._nav = NavController(self)

        # ── Central widget: sidebar + content ─────────────────
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Title bar
        self._title_bar = _TitleBar()
        outer.addWidget(self._title_bar)

        # Top action bar (New/Open/Save/Import/Export/OCR/Translate/Inpaint/Typeset)
        self._action_bar = TopActionBar()
        self._action_bar.new_project_clicked.connect(self._new_project)
        self._action_bar.open_project_clicked.connect(lambda: self._open_project(""))
        self._action_bar.save_clicked.connect(self._save_project)
        self._action_bar.import_clicked.connect(self._import_project)
        self._action_bar.export_clicked.connect(self._export_project)
        self._action_bar.ocr_clicked.connect(self._run_ocr)
        self._action_bar.translate_clicked.connect(self._run_translate)
        self._action_bar.inpaint_clicked.connect(self._run_inpaint)
        self._action_bar.typeset_clicked.connect(self._run_typeset)
        outer.addWidget(self._action_bar)

        # Main row: sidebar + stack
        main_row = QHBoxLayout()
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(0)

        # Sidebar (QML first, QWidget fallback)
        self._sidebar = self._create_sidebar()
        main_row.addWidget(self._sidebar)

        # Content stack
        self._stack = QStackedWidget()
        self._stack.setObjectName("ContentStack")
        main_row.addWidget(self._stack, 1)

        outer.addLayout(main_row, 1)

        # ── Create pages ──────────────────────────────────────
        self._pages: dict[str, QWidget] = {}
        self._create_pages()

        # Status footer
        self._status_footer = StatusFooter()
        outer.addWidget(self._status_footer)

        # Job Status Drawer (hidden by default, Ctrl+J toggles)
        self._job_drawer = JobStatusDrawer()
        self._job_drawer.hide()
        outer.addWidget(self._job_drawer)

        # Command Palette (Ctrl+K)
        self._command_palette = CommandPalette(self)
        self._command_palette.set_commands([
            ("Go to Home", "nav:home"),
            ("Go to Editor", "nav:editor"),
            ("Open Live Translate", "nav:live_translate"),
            ("Open Quick Image", "nav:quick_image"),
            ("Open Downloader", "nav:downloader"),
            ("Open Batch Queue", "nav:batch_queue"),
            ("Open Assist / QA", "nav:assist_qa"),
            ("Open Models / AI", "nav:models_ai"),
            ("Open Settings", "nav:settings"),
            ("Open Diagnostics", "nav:diagnostics"),
            ("Toggle Job Status Drawer", "toggle:jobs"),
        ])
        self._command_palette.command_selected.connect(self._run_shell_command)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self._command_palette.show)
        QShortcut(QKeySequence("Ctrl+J"), self, activated=self._toggle_job_drawer)

        # Common action shortcuts
        QShortcut(QKeySequence("Ctrl+N"), self, activated=self._new_project)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=lambda: self._open_project(""))
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_project)

        # ── Wire navigation ───────────────────────────────────
        self._nav.sectionChanged.connect(self._on_navigate)

        # ── Detect system info ────────────────────────────────
        QTimer.singleShot(500, self._detect_system_info)

        # ── Auto-open project if requested ────────────────────
        if open_dir:
            QTimer.singleShot(100, lambda: self._open_project(open_dir))

    def _create_sidebar(self) -> QWidget:
        qml_path = osp.join(osp.dirname(__file__), "qml", "Sidebar.qml")
        try:
            sidebar = QQuickWidget()
            sidebar.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
            sidebar.setClearColor(Qt.GlobalColor.transparent)
            sidebar.rootContext().setContextProperty("navController", self._nav)
            sidebar.setSource(QUrl.fromLocalFile(qml_path))
            if sidebar.status() == QQuickWidget.Status.Error:
                errors = "; ".join(str(err.toString()) for err in sidebar.errors())
                raise RuntimeError(f"Sidebar.qml failed to load: {errors}")
            root = sidebar.rootObject()
            if root is None:
                raise RuntimeError("Sidebar.qml rootObject() is None")
            root.navigateRequested.connect(self._nav.navigate)
            self._nav.sectionChanged.connect(lambda section: root.setProperty("currentSection", section))
            root.setProperty("currentSection", self._nav.currentSection)
            sidebar.setFixedWidth(200)
            return sidebar
        except Exception as exc:
            print(f"[BallonsShell] Falling back to QWidget sidebar: {exc}")
            return SidebarWidget(self._nav)

    def _create_pages(self):
        """Instantiate all section pages and add to stack."""
        # HOME – full implementation
        home = HomePage()
        home.open_project_requested.connect(self._open_project)
        home.new_project_requested.connect(self._new_project)
        home.navigate_requested.connect(self._nav.navigate)
        self._add_page("home", home)

        # Populate recent projects from config
        self._refresh_recent_projects()

        # EDITOR – redesigned workspace shell
        self._add_page("editor", EditorPage())

        # Core redesigned section shells
        self._add_page("live_translate", LiveTranslatePage())
        self._add_page("quick_image", QuickImagePage())
        self._add_page("downloader", DownloaderPage())
        self._add_page("batch_queue", BatchQueuePage())
        self._add_page("assist_qa", AssistQAPage())
        self._add_page("models_ai", ModelsAIPage())
        self._add_page("settings", SettingsPage())
        self._add_page("diagnostics", DiagnosticsPage())

        # All other sections – stubs for now
        for section_id in SECTIONS:
            if section_id in {"home", "editor", "live_translate", "quick_image", "downloader", "batch_queue", "assist_qa", "models_ai", "settings", "diagnostics"}:
                continue
            page = StubPage(section_id)
            self._add_page(section_id, page)

    def _add_page(self, section_id: str, widget: QWidget):
        idx = self._stack.addWidget(widget)
        self._pages[section_id] = widget

    def _on_navigate(self, section_id: str):
        page = self._pages.get(section_id)
        if page:
            self._stack.setCurrentWidget(page)

    def _toggle_job_drawer(self):
        self._job_drawer.setVisible(not self._job_drawer.isVisible())

    def _run_shell_command(self, command_id: str):
        if command_id.startswith("nav:"):
            self._nav.navigate(command_id.split(":", 1)[1])
        elif command_id == "toggle:jobs":
            self._toggle_job_drawer()

    # ── Project actions ───────────────────────────────────────
    def _open_project(self, path: str = ""):
        if not path:
            path = QFileDialog.getExistingDirectory(
                self, "Open Project Folder", "",
                QFileDialog.Option.ShowDirsOnly,
            )
        if path:
            self._title_bar.set_project(osp.basename(path))
            self._status_footer.project_label.setText(osp.basename(path))
            self._status_footer.status_label.setText("Project loaded")
            # TODO: wire to actual project loading

    def _new_project(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Folder for New Project", "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if path:
            self._title_bar.set_project(osp.basename(path))
            self._status_footer.project_label.setText(osp.basename(path))
            # TODO: wire to actual project creation

    def _save_project(self):
        # TODO: wire to actual save
        self._status_footer.status_label.setText("Saved")

    def _import_project(self):
        path = QFileDialog.getExistingDirectory(
            self, "Import Project Folder", "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if path:
            self._title_bar.set_project(osp.basename(path))
            self._status_footer.project_label.setText(osp.basename(path))

    def _export_project(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Export Destination", "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if path:
            self._status_footer.status_label.setText(f"Exported to {osp.basename(path)}")

    def _run_ocr(self):
        self._status_footer.status_label.setText("Running OCR...")
        # TODO: wire to actual OCR pipeline

    def _run_translate(self):
        self._status_footer.status_label.setText("Running Translation...")
        # TODO: wire to actual translation pipeline

    def _run_inpaint(self):
        self._status_footer.status_label.setText("Running Inpaint...")
        # TODO: wire to actual inpaint pipeline

    def _run_typeset(self):
        self._status_footer.status_label.setText("Running Typeset...")
        # TODO: wire to actual typeset pipeline

    def _refresh_recent_projects(self):
        """Pull recent projects from config and push to HomePage."""
        recent = getattr(self._config, 'recent_proj_list', []) or []
        items = []
        for p in recent[:8]:
            name = osp.basename(str(p).rstrip("/\\"))
            items.append({"name": name, "path": str(p), "date": ""})
        home = self._pages.get("home")
        if home and isinstance(home, HomePage):
            home.set_recent_projects(items)

    # ── System info ───────────────────────────────────────────
    def _detect_system_info(self):
        gpu_text = ""
        vram_text = ""
        ram_text = ""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_text = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_mem
                vram_text = f"{total / (1024**3):.1f} GB"
        except Exception:
            gpu_text = "N/A"

        try:
            import psutil
            mem = psutil.virtual_memory()
            ram_text = f"{mem.total / (1024**3):.1f} GB"
        except Exception:
            ram_text = "N/A"

        self._title_bar.set_gpu_info(f"GPU: {gpu_text}" if gpu_text else "")
        self._title_bar.set_ram_info(f"RAM: {ram_text}" if ram_text else "")

        home = self._pages.get("home")
        if home and isinstance(home, HomePage):
            home.set_system_status(gpu=gpu_text, vram=vram_text, ram=ram_text)

    # ── Close ─────────────────────────────────────────────────
    def closeEvent(self, event: QCloseEvent):
        try:
            from utils.config import save_config
            save_config()
        except Exception:
            pass
        super().closeEvent(event)
