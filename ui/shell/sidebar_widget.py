"""
Sidebar navigation widget – pure QWidget implementation matching the mockup.

Vertical icon+label list with active state highlighting, matching the
dark-purple mockup design.
"""

from __future__ import annotations
import os.path as osp
from typing import Optional

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QSizePolicy, QSpacerItem,
)
from qtpy.QtCore import Qt, Signal, QSize
from qtpy.QtGui import QIcon, QFont, QPixmap, QPainter, QColor

from .nav_controller import SECTIONS, SECTION_LABELS, NavController
from .theme import COLORS, SPACING, FONTS, SIDEBAR_WIDTH, build_sidebar_qss


class SidebarButton(QPushButton):
    """Single nav item in the sidebar."""

    def __init__(self, section_id: str, label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.section_id = section_id
        self.setObjectName("SidebarButton")
        self.setText(label)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setCheckable(True)
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Dynamic property for QSS active state
        self.setProperty("active", False)

    def set_active(self, active: bool):
        self.setProperty("active", active)
        self.setChecked(active)
        self.style().unpolish(self)
        self.style().polish(self)


class SidebarSeparator(QFrame):
    """Thin horizontal line between sidebar sections."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("SidebarSeparator")
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(1)


class SidebarWidget(QWidget):
    """Full sidebar widget with logo, navigation buttons, and bottom actions."""

    navigate_requested = Signal(str)  # section_id

    def __init__(self, nav: NavController, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("SidebarContainer")
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setStyleSheet(build_sidebar_qss())
        self._nav = nav
        self._buttons: dict[str, SidebarButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING.sm, SPACING.md, SPACING.sm, SPACING.md)
        layout.setSpacing(2)

        # ── Logo / brand ──────────────────────────────────────
        logo = QLabel("B")
        logo.setObjectName("SidebarLogo")
        logo.setFixedHeight(48)
        logo.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(logo)

        layout.addWidget(SidebarSeparator())
        layout.addSpacing(SPACING.sm)

        # ── Main nav buttons ──────────────────────────────────
        # Top group: Home, Editor
        for sid in ["home", "editor"]:
            btn = self._make_button(sid)
            layout.addWidget(btn)

        layout.addWidget(SidebarSeparator())
        layout.addSpacing(SPACING.xs)

        # Middle group: feature pages
        for sid in ["live_translate", "quick_image", "downloader", "batch_queue", "assist_qa", "models_ai"]:
            btn = self._make_button(sid)
            layout.addWidget(btn)

        layout.addStretch(1)

        # Bottom group: Settings, Diagnostics
        layout.addWidget(SidebarSeparator())
        layout.addSpacing(SPACING.xs)
        for sid in ["settings", "diagnostics"]:
            btn = self._make_button(sid)
            layout.addWidget(btn)

        # ── Wire up nav controller ────────────────────────────
        nav.sectionChanged.connect(self._on_section_changed)
        self._on_section_changed(nav.currentSection)

    def _make_button(self, section_id: str) -> SidebarButton:
        label = SECTION_LABELS.get(section_id, section_id)
        btn = SidebarButton(section_id, label, self)
        btn.clicked.connect(lambda checked, sid=section_id: self._on_clicked(sid))
        self._buttons[section_id] = btn
        return btn

    def _on_clicked(self, section_id: str):
        self._nav.navigate(section_id)
        self.navigate_requested.emit(section_id)

    def _on_section_changed(self, section_id: str):
        for sid, btn in self._buttons.items():
            btn.set_active(sid == section_id)
