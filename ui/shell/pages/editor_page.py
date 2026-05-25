"""
Editor / Workspace page for the new shell.

This page provides the redesigned workspace chrome: page tabs, central canvas
surface, tool rail, right inspector, thumbnails, and bottom job/status drawer.
The real legacy Canvas integration is intentionally isolated for later because
Canvas currently depends on MainWindow callbacks and global shared widgets.
"""

from __future__ import annotations
from typing import Optional

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QSplitter, QListWidget, QListWidgetItem, QProgressBar,
    QTabWidget, QComboBox, QCheckBox, QPlainTextEdit, QSizePolicy,
)
from qtpy.QtCore import Qt

from ..theme import COLORS, FONTS, SPACING, RADIUS


class _Panel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("EditorPanel")
        self.setStyleSheet(f"""
            #EditorPanel {{
                background-color: {COLORS.bg_surface};
                border: 1px solid {COLORS.border};
                border-radius: {RADIUS.md}px;
            }}
        """)


class _ToolButton(QPushButton):
    def __init__(self, text: str, active: bool = False, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(36, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("accent", active)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS.accent if active else COLORS.bg_surface};
                color: {COLORS.text_inverse if active else COLORS.text_secondary};
                border: 1px solid {COLORS.border};
                border-radius: {RADIUS.md}px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                border-color: {COLORS.accent};
                color: {COLORS.text_primary};
            }}
        """)


class EditorPage(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.lg, SPACING.lg, SPACING.lg, SPACING.lg)
        root.setSpacing(SPACING.md)

        root.addLayout(self._build_top_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_toolrail())
        splitter.addWidget(self._build_canvas_area())
        splitter.addWidget(self._build_thumbnail_strip())
        splitter.addWidget(self._build_inspector())
        splitter.setSizes([48, 850, 84, 280])
        root.addWidget(splitter, 1)

        root.addWidget(self._build_job_drawer())

    def _build_top_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(SPACING.sm)

        tab = QLabel("page_001.png  ×")
        tab.setStyleSheet(f"""
            background-color: {COLORS.bg_surface};
            color: {COLORS.text_primary};
            border: 1px solid {COLORS.border};
            border-bottom: 2px solid {COLORS.accent};
            border-radius: {RADIUS.sm}px;
            padding: 7px 14px;
            font-weight: 600;
        """)
        bar.addWidget(tab)
        bar.addStretch()

        for label in ["100%", "Fit", "Grid", "OCR", "Translate", "Inpaint"]:
            btn = QPushButton(label)
            if label in {"OCR", "Translate", "Inpaint"}:
                btn.setProperty("accent", True)
            bar.addWidget(btn)
        return bar

    def _build_left_toolrail(self) -> QWidget:
        rail = _Panel()
        rail.setFixedWidth(52)
        lay = QVBoxLayout(rail)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        tools = [("✋", True), ("▣", False), ("T", False), ("✎", False), ("⌫", False), ("⌕", False)]
        for txt, active in tools:
            lay.addWidget(_ToolButton(txt, active))
        lay.addStretch()
        return rail

    def _build_canvas_area(self) -> QWidget:
        wrapper = _Panel()
        lay = QVBoxLayout(wrapper)
        lay.setContentsMargins(SPACING.lg, SPACING.lg, SPACING.lg, SPACING.lg)
        lay.setSpacing(SPACING.md)

        canvas = QFrame()
        canvas.setObjectName("MockCanvas")
        canvas.setMinimumSize(520, 360)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas.setStyleSheet(f"""
            #MockCanvas {{
                background-color: #F4F4F6;
                border: 1px solid {COLORS.border};
                border-radius: {RADIUS.md}px;
            }}
        """)
        c_lay = QVBoxLayout(canvas)
        c_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel("Editor canvas placeholder\nLegacy Canvas integration comes next")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #222233; font-size: 18px; font-weight: 700; background: transparent;")
        c_lay.addWidget(label)
        lay.addWidget(canvas, 1)
        return wrapper

    def _build_thumbnail_strip(self) -> QWidget:
        panel = _Panel()
        panel.setFixedWidth(90)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        for i in range(1, 7):
            thumb = QLabel(f"{i:03}")
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb.setFixedSize(62, 72)
            thumb.setStyleSheet(f"""
                background-color: {COLORS.bg_base};
                color: {COLORS.text_secondary};
                border: 1px solid {COLORS.accent if i == 1 else COLORS.border};
                border-radius: {RADIUS.sm}px;
                font-weight: 700;
            """)
            lay.addWidget(thumb)
        lay.addStretch()
        return panel

    def _build_inspector(self) -> QWidget:
        panel = _Panel()
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(340)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(SPACING.md, SPACING.md, SPACING.md, SPACING.md)
        lay.setSpacing(SPACING.md)

        title = QLabel("Inspector")
        title.setStyleSheet(f"color: {COLORS.text_primary}; font-size: {FONTS.size_lg}px; font-weight: 700; background: transparent;")
        lay.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(self._build_ocr_tab(), "OCR")
        tabs.addTab(self._build_text_tab(), "Text")
        tabs.addTab(self._build_style_tab(), "Style")
        tabs.addTab(self._build_layer_tab(), "Layer")
        lay.addWidget(tabs, 1)
        return panel

    def _build_ocr_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        for label, values in [
            ("Language", ["English", "Japanese", "Auto Detect"]),
            ("Mode", ["Manga", "Comic", "Document"]),
            ("Engine", ["Default", "OCR 2x", "OCR 4x"]),
        ]:
            lay.addWidget(QLabel(label))
            cb = QComboBox()
            cb.addItems(values)
            lay.addWidget(cb)
        chk = QCheckBox("Disable Vertical Text")
        lay.addWidget(chk)
        run = QPushButton("Run OCR")
        run.setProperty("accent", True)
        lay.addWidget(run)
        lay.addStretch()
        return w

    def _build_text_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Source Text"))
        src = QPlainTextEdit()
        src.setPlainText("Source text appears here...")
        lay.addWidget(src)
        lay.addWidget(QLabel("Translation"))
        tr = QPlainTextEdit()
        tr.setPlainText("Translation appears here...")
        lay.addWidget(tr)
        return w

    def _build_style_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        for label, values in [
            ("Font", ["Anime Ace", "Noto Sans", "Arial"]),
            ("Size", ["18", "24", "32"]),
            ("Alignment", ["Center", "Left", "Right"]),
        ]:
            lay.addWidget(QLabel(label))
            cb = QComboBox()
            cb.addItems(values)
            lay.addWidget(cb)
        lay.addStretch()
        return w

    def _build_layer_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        for name in ["Original", "Inpaint", "Text", "Guides"]:
            lay.addWidget(QCheckBox(name))
        lay.addStretch()
        return w

    def _build_job_drawer(self) -> QWidget:
        panel = _Panel()
        panel.setFixedHeight(120)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.sm)
        lay.setSpacing(SPACING.sm)
        header = QHBoxLayout()
        title = QLabel("Job Status")
        title.setStyleSheet(f"color: {COLORS.text_primary}; font-weight: 700; background: transparent;")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(QLabel("Overall Progress"))
        progress = QProgressBar()
        progress.setValue(55)
        progress.setFixedWidth(180)
        header.addWidget(progress)
        lay.addLayout(header)

        grid = QGridLayout()
        rows = [("OCR", "Running", 72), ("Translate", "Queued", 18), ("Inpaint", "Idle", 0)]
        for r, (stage, status, pct) in enumerate(rows):
            grid.addWidget(QLabel(stage), r, 0)
            grid.addWidget(QLabel(status), r, 1)
            pb = QProgressBar()
            pb.setValue(pct)
            grid.addWidget(pb, r, 2)
        lay.addLayout(grid)
        return panel
