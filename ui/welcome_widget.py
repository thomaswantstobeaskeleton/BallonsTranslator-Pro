# -*- coding: utf-8 -*-
"""
Welcome / first-start screen when no project is open.
Inspired by manhua-translator and Komakun: select or create project from a single, bubbly welcome view.
"""
import os.path as osp

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
    QFileDialog,
    QSizePolicy,
)
from qtpy.QtCore import Qt, Signal, QTimer

from .custom_widget import Widget
from .custom_widget.push_button import NoBorderPushBtn
from .custom_widget.helper import isDarkTheme


def _welcome_stylesheet(dark: bool) -> str:
    """Build a single cohesive stylesheet for the welcome screen (light or dark)."""
    if dark:
        # Dark: align with eva-dark (#282c34, #21252b, #535671)
        bg_gradient = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #25282e, stop:1 #1e2126)"
        card_bg = "#2c3036"
        card_border = "rgba(255, 255, 255, 0.06)"
        title_color = "#c8d0e0"
        subtitle_color = "#8e99b1"
        card_title_color = "#9ca3af"
        placeholder_color = "#6b7280"
        recent_btn_bg = "rgba(255, 255, 255, 0.06)"
        recent_btn_bg_hover = "rgba(255, 255, 255, 0.12)"
        recent_btn_border = "rgba(91, 143, 249, 0.4)"
    else:
        # Light: soft greys and blue-grey typography
        bg_gradient = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #eef2f7, stop:1 #e2e8f0)"
        card_bg = "rgba(255, 255, 255, 0.85)"
        card_border = "rgba(0, 0, 0, 0.06)"
        title_color = "#1e3a5f"
        subtitle_color = "#64748b"
        card_title_color = "#475569"
        placeholder_color = "#94a3b8"
        recent_btn_bg = "rgba(0, 0, 0, 0.04)"
        recent_btn_bg_hover = "rgba(0, 0, 0, 0.08)"
        recent_btn_border = "rgba(59, 130, 246, 0.5)"

    return f"""
        WelcomeWidget#WelcomeWidget {{
            background: {bg_gradient};
        }}
        WelcomeWidget WelcomeCard#WelcomeCard {{
            background-color: {card_bg};
            border-radius: 16px;
            border: 1px solid {card_border};
        }}
        WelcomeWidget QLabel#WelcomeTitle {{
            font-size: 28px;
            font-weight: bold;
            color: {title_color};
        }}
        WelcomeWidget QLabel#WelcomeSubtitle {{
            font-size: 14px;
            color: {subtitle_color};
        }}
        WelcomeWidget QLabel#WelcomeCardTitle {{
            font-weight: bold;
            font-size: 13px;
            color: {card_title_color};
        }}
        WelcomeWidget QLabel#WelcomeRecentPlaceholder {{
            color: {placeholder_color};
            font-size: 12px;
        }}
        WelcomeWidget QScrollArea {{
            background: transparent;
        }}
        /* Recent project buttons only (inside scroll area): subtle surface and soft focus */
        WelcomeWidget QScrollArea QPushButton {{
            background-color: {recent_btn_bg};
            border: 1px solid transparent;
            border-radius: 10px;
            text-align: left;
            padding: 8px 12px;
        }}
        WelcomeWidget QScrollArea QPushButton:hover {{
            background-color: {recent_btn_bg_hover};
        }}
        WelcomeWidget QScrollArea QPushButton:focus {{
            border: 1px solid {recent_btn_border};
        }}
    """


class WelcomeCard(QFrame):
    """Bubbly card container for a section (actions or recent list)."""
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("WelcomeCard")
        # Styling is applied by parent WelcomeWidget via _welcome_stylesheet()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        if title:
            title_lbl = QLabel(title)
            title_lbl.setObjectName("WelcomeCardTitle")
            layout.addWidget(title_lbl)
        self._layout = layout

    def addWidget(self, w):
        self._layout.addWidget(w)

    def addLayout(self, l):
        self._layout.addLayout(l)

    def addStretch(self):
        self._layout.addStretch()


class WelcomeWidget(Widget):
    """First-start / welcome screen: open or create project, recent projects list."""
    open_folder_requested = Signal()
    open_project_requested = Signal(str)
    open_images_requested = Signal()
    open_acbf_requested = Signal()
    recent_project_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("WelcomeWidget")
        self._recent_buttons = []
        self._build_ui()
        self._update_styles()

    def _update_styles(self):
        """Apply cohesive light/dark styles from current theme."""
        self.setStyleSheet(_welcome_stylesheet(isDarkTheme()))

    def showEvent(self, event):
        super().showEvent(event)
        # Defer stylesheet update to avoid QPainter conflicts with QGraphicsOpacityEffect
        # during the show/paint cycle (fixes "paint device can only be painted by one painter" spam).
        QTimer.singleShot(0, self._update_styles)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(24)
        layout.setContentsMargins(48, 48, 48, 48)

        # Title
        title = QLabel(self.tr("BallonsTranslator Pro"))
        title.setObjectName("WelcomeTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel(self.tr("Open a project folder or choose a recent one to start."))
        subtitle.setObjectName("WelcomeSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(16)

        # Primary actions card
        actions_card = WelcomeCard(self.tr("Get started"), self)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self._btn_open_folder = NoBorderPushBtn(self.tr("Open folder…"))
        self._btn_open_folder.setToolTip(self.tr("Open a folder containing images (creates or loads project)."))
        self._btn_open_folder.setMinimumHeight(44)
        self._btn_open_folder.setMinimumWidth(160)
        self._btn_open_folder.clicked.connect(self._on_open_folder)
        btn_layout.addWidget(self._btn_open_folder)

        self._btn_open_project = NoBorderPushBtn(self.tr("Open project…"))
        self._btn_open_project.setToolTip(self.tr("Open an existing project (.json)."))
        self._btn_open_project.setMinimumHeight(44)
        self._btn_open_project.setMinimumWidth(160)
        self._btn_open_project.clicked.connect(self._on_open_project)
        btn_layout.addWidget(self._btn_open_project)

        self._btn_open_images = NoBorderPushBtn(self.tr("Open images…"))
        self._btn_open_images.setToolTip(self.tr("Select image files to open as a new project."))
        self._btn_open_images.setMinimumHeight(44)
        self._btn_open_images.setMinimumWidth(140)
        self._btn_open_images.clicked.connect(self.open_images_requested.emit)
        btn_layout.addWidget(self._btn_open_images)

        self._btn_open_acbf = NoBorderPushBtn(self.tr("Open ACBF/CBZ…"))
        self._btn_open_acbf.setToolTip(self.tr("Open a comic archive (.cbz/.zip)."))
        self._btn_open_acbf.setMinimumHeight(44)
        self._btn_open_acbf.setMinimumWidth(140)
        self._btn_open_acbf.clicked.connect(self.open_acbf_requested.emit)
        btn_layout.addWidget(self._btn_open_acbf)

        btn_layout.addStretch()
        actions_card.addLayout(btn_layout)
        layout.addWidget(actions_card)

        # Recent projects card
        self._recent_card = WelcomeCard(self.tr("Recent projects"), self)
        self._recent_scroll = QScrollArea()
        self._recent_scroll.setWidgetResizable(True)
        self._recent_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._recent_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._recent_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        self._recent_inner = QWidget()
        self._recent_inner_layout = QVBoxLayout(self._recent_inner)
        self._recent_inner_layout.setContentsMargins(0, 0, 8, 0)
        self._recent_inner_layout.setSpacing(6)
        self._recent_placeholder = QLabel(self.tr("No recent projects."))
        self._recent_placeholder.setObjectName("WelcomeRecentPlaceholder")
        self._recent_inner_layout.addWidget(self._recent_placeholder)
        self._recent_scroll.setWidget(self._recent_inner)
        self._recent_card.addWidget(self._recent_scroll)
        self._recent_scroll.setMinimumHeight(120)
        self._recent_scroll.setMaximumHeight(220)
        layout.addWidget(self._recent_card)

        layout.addStretch()

    def _on_open_folder(self):
        self.open_folder_requested.emit()

    def _on_open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open project"),
            "",
            self.tr("Project files (*.json);;All files (*)"),
        )
        if path and osp.isfile(path):
            self.open_project_requested.emit(path)

    def set_recent_projects(self, paths: list):
        """Update the list of recent project paths (full directory or .json paths)."""
        for btn in self._recent_buttons:
            btn.deleteLater()
        self._recent_buttons.clear()

        self._recent_placeholder.setVisible(len(paths) == 0)
        for p in paths:
            if not p or not osp.exists(p):
                continue
            display = osp.basename(osp.dirname(p)) if p.endswith(".json") else osp.basename(p)
            if not display:
                display = p
            btn = NoBorderPushBtn(display)
            btn.setToolTip(p)
            btn.setMinimumHeight(36)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda checked=False, path=p: self.recent_project_clicked.emit(path))
            self._recent_inner_layout.insertWidget(self._recent_inner_layout.count() - 1, btn)
            self._recent_buttons.append(btn)
