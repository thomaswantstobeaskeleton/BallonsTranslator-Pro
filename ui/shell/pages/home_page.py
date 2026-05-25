"""
HOME / DASHBOARD  –  Default landing page (mockup section 1).

Shows:
  - Welcome header
  - Recent Projects list
  - Quick Actions grid (Open Project, Quick OCR, Translate Folder, etc.)
  - System Status (GPU, VRAM, Disk, Providers)
"""

from __future__ import annotations
import os, os.path as osp, time
from typing import Optional, List, Callable

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy, QSpacerItem,
)
from qtpy.QtCore import Qt, Signal, QSize
from qtpy.QtGui import QFont, QColor, QIcon

from ..theme import COLORS, SPACING, FONTS, RADIUS


# ── Card helper ───────────────────────────────────────────────
class _Card(QFrame):
    """Rounded card container matching the mockup surface color."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DashboardCard")
        self.setStyleSheet(f"""
            #DashboardCard {{
                background-color: {COLORS.bg_surface};
                border: 1px solid {COLORS.border};
                border-radius: {RADIUS.lg}px;
                padding: {SPACING.lg}px;
            }}
        """)


class _QuickActionButton(QPushButton):
    """Large icon+label button for the Quick Actions grid."""

    def __init__(self, label: str, icon_char: str = "", parent=None):
        super().__init__(parent)
        self.setText(label)
        self.setMinimumSize(120, 90)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("QuickActionBtn")
        self.setStyleSheet(f"""
            #QuickActionBtn {{
                background-color: {COLORS.bg_surface};
                color: {COLORS.text_primary};
                border: 1px solid {COLORS.border};
                border-radius: {RADIUS.lg}px;
                padding: {SPACING.md}px;
                font-size: {FONTS.size_sm}px;
                font-weight: 500;
            }}
            #QuickActionBtn:hover {{
                background-color: {COLORS.bg_surface_hover};
                border-color: {COLORS.border_focus};
            }}
        """)


class _RecentProjectItem(QFrame):
    """Single row in the Recent Projects list."""

    clicked = Signal(str)  # project path

    def __init__(self, name: str, path: str, date_str: str, parent=None):
        super().__init__(parent)
        self._path = path
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("RecentProjectItem")
        self.setStyleSheet(f"""
            #RecentProjectItem {{
                background: transparent;
                border-radius: {RADIUS.md}px;
                padding: {SPACING.sm}px {SPACING.md}px;
            }}
            #RecentProjectItem:hover {{
                background-color: {COLORS.bg_surface_hover};
            }}
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(name)
        lbl.setStyleSheet(f"color: {COLORS.text_primary}; font-weight: 500; background: transparent;")
        date_lbl = QLabel(date_str)
        date_lbl.setStyleSheet(f"color: {COLORS.text_muted}; font-size: {FONTS.size_xs}px; background: transparent;")
        date_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(lbl, 1)
        lay.addWidget(date_lbl)

    def mousePressEvent(self, event):
        self.clicked.emit(self._path)
        super().mousePressEvent(event)


# ── Status pill ───────────────────────────────────────────────
class _StatusPill(QLabel):
    def __init__(self, text: str, color: str = COLORS.success, parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"""
            background-color: {color};
            color: {COLORS.text_inverse};
            border-radius: 10px;
            padding: 2px 10px;
            font-size: {FONTS.size_xs}px;
            font-weight: 600;
        """)
        self.setFixedHeight(20)


# ── Main page ─────────────────────────────────────────────────
class HomePage(QWidget):
    """Dashboard landing page."""

    open_project_requested = Signal(str)    # path
    new_project_requested = Signal()
    navigate_requested = Signal(str)        # section_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._recent_projects: List[dict] = []  # [{name, path, date}]
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        root.setSpacing(SPACING.xl)

        # ── Welcome header ────────────────────────────────────
        welcome = QLabel("Welcome back!")
        welcome.setStyleSheet(f"color: {COLORS.text_primary}; font-size: {FONTS.size_h1}px; font-weight: 700; background: transparent;")
        subtitle = QLabel("What would you like to work on today?")
        subtitle.setStyleSheet(f"color: {COLORS.text_secondary}; font-size: {FONTS.size_base}px; background: transparent;")
        root.addWidget(welcome)
        root.addWidget(subtitle)
        root.addSpacing(SPACING.sm)

        # ── Content row: Quick Actions + Recent Projects ──────
        content_row = QHBoxLayout()
        content_row.setSpacing(SPACING.xl)

        # Quick Actions card
        qa_card = _Card()
        qa_lay = QVBoxLayout(qa_card)
        qa_title = QLabel("Quick Actions")
        qa_title.setStyleSheet(f"color: {COLORS.text_primary}; font-size: {FONTS.size_lg}px; font-weight: 600; background: transparent;")
        qa_lay.addWidget(qa_title)
        qa_lay.addSpacing(SPACING.sm)

        qa_grid = QGridLayout()
        qa_grid.setSpacing(SPACING.md)

        actions = [
            ("Open Project", "open_project"),
            ("New Project", "new_project"),
            ("Quick OCR", "quick_image"),
            ("Translate Folder", "batch_queue"),
            ("Batch Queue", "batch_queue"),
            ("Translation Assist", "assist_qa"),
        ]
        for i, (label, action_id) in enumerate(actions):
            btn = _QuickActionButton(label)
            btn.clicked.connect(lambda checked, a=action_id: self._on_quick_action(a))
            qa_grid.addWidget(btn, i // 3, i % 3)

        qa_lay.addLayout(qa_grid)
        qa_lay.addStretch()
        content_row.addWidget(qa_card, 3)

        # Recent Projects card
        rp_card = _Card()
        rp_lay = QVBoxLayout(rp_card)
        rp_header = QHBoxLayout()
        rp_title = QLabel("Recent Projects")
        rp_title.setStyleSheet(f"color: {COLORS.text_primary}; font-size: {FONTS.size_lg}px; font-weight: 600; background: transparent;")
        self._view_all_btn = QPushButton("View All")
        self._view_all_btn.setObjectName("LinkButton")
        self._view_all_btn.setStyleSheet(f"""
            #LinkButton {{
                background: transparent;
                color: {COLORS.accent};
                border: none;
                font-size: {FONTS.size_sm}px;
                font-weight: 500;
            }}
            #LinkButton:hover {{ color: {COLORS.accent_hover}; }}
        """)
        self._view_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        rp_header.addWidget(rp_title)
        rp_header.addStretch()
        rp_header.addWidget(self._view_all_btn)
        rp_lay.addLayout(rp_header)

        self._recent_list_layout = QVBoxLayout()
        self._recent_list_layout.setSpacing(2)
        rp_lay.addLayout(self._recent_list_layout)
        rp_lay.addStretch()
        content_row.addWidget(rp_card, 2)

        root.addLayout(content_row)

        # ── System Status row ─────────────────────────────────
        status_card = _Card()
        status_lay = QHBoxLayout(status_card)
        status_title = QLabel("System Status")
        status_title.setStyleSheet(f"color: {COLORS.text_primary}; font-size: {FONTS.size_lg}px; font-weight: 600; background: transparent;")
        status_lay.addWidget(status_title)
        status_lay.addSpacing(SPACING.xl)

        self._status_gpu = QLabel("GPU: detecting...")
        self._status_gpu.setStyleSheet(f"color: {COLORS.text_secondary}; background: transparent;")
        self._status_vram = QLabel("VRAM: --")
        self._status_vram.setStyleSheet(f"color: {COLORS.text_secondary}; background: transparent;")
        self._status_ram = QLabel("RAM: --")
        self._status_ram.setStyleSheet(f"color: {COLORS.text_secondary}; background: transparent;")

        status_lay.addWidget(self._status_gpu)
        status_lay.addWidget(self._status_vram)
        status_lay.addWidget(self._status_ram)
        status_lay.addStretch()

        self._status_pill = _StatusPill("All Systems Operational")
        status_lay.addWidget(self._status_pill)

        root.addWidget(status_card)
        root.addStretch()

        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Public API ────────────────────────────────────────────
    def set_recent_projects(self, projects: List[dict]):
        """projects: list of {name, path, date}"""
        self._recent_projects = projects
        # Clear existing
        while self._recent_list_layout.count():
            item = self._recent_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Add new
        for proj in projects[:8]:
            item = _RecentProjectItem(
                proj.get("name", "Untitled"),
                proj.get("path", ""),
                proj.get("date", ""),
            )
            item.clicked.connect(self.open_project_requested.emit)
            self._recent_list_layout.addWidget(item)

    def set_system_status(self, gpu: str = "", vram: str = "", ram: str = ""):
        if gpu:
            self._status_gpu.setText(f"GPU: {gpu}")
        if vram:
            self._status_vram.setText(f"VRAM: {vram}")
        if ram:
            self._status_ram.setText(f"RAM: {ram}")

    # ── Internal ──────────────────────────────────────────────
    def _on_quick_action(self, action_id: str):
        if action_id == "open_project":
            self.open_project_requested.emit("")
        elif action_id == "new_project":
            self.new_project_requested.emit()
        else:
            self.navigate_requested.emit(action_id)
