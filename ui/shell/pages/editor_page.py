"""
Editor / Workspace page for the new shell.

Functional editor workspace that can open projects, display images,
and run OCR/Translate/Inpaint pipeline stages.
"""

from __future__ import annotations
from typing import Optional, List
import os
import os.path as osp

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QSplitter, QListWidget, QListWidgetItem, QProgressBar,
    QTabWidget, QComboBox, QCheckBox, QPlainTextEdit, QSizePolicy,
    QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QMessageBox, QLineEdit,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QPixmap, QImage, QPainter, QColor, QFont

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
    """Functional editor page with image viewer and pipeline controls."""

    ocr_requested = Signal()
    translate_requested = Signal()
    inpaint_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._project_path: Optional[str] = None
        self._image_files: List[str] = []
        self._current_index: int = -1
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
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

    # ── Public API ──────────────────────────────────────────
    def open_project(self, path: str):
        """Open a project folder and load image list."""
        if not osp.isdir(path):
            return
        self._project_path = path
        self._image_files = []
        # Look for image files in project
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
            if osp.isdir(osp.join(path, "orig")):
                d = osp.join(path, "orig")
            else:
                d = path
            for f in sorted(os.listdir(d) if osp.isdir(d) else []):
                if f.lower().endswith(ext):
                    self._image_files.append(osp.join(d, f))
        self._current_index = 0 if self._image_files else -1
        self._refresh_thumbnails()
        self._load_current_image()
        self._update_tab_label()

    def open_image(self, path: str):
        """Open a single image."""
        if osp.isfile(path):
            self._image_files = [path]
            self._project_path = osp.dirname(path)
            self._current_index = 0
            self._refresh_thumbnails()
            self._load_current_image()
            self._update_tab_label()

    def current_image_path(self) -> Optional[str]:
        if 0 <= self._current_index < len(self._image_files):
            return self._image_files[self._current_index]
        return None

    def _build_top_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(SPACING.sm)

        self._tab_label = QLabel("No image open")
        self._tab_label.setStyleSheet(f"""
            background-color: {COLORS.bg_surface};
            color: {COLORS.text_primary};
            border: 1px solid {COLORS.border};
            border-bottom: 2px solid {COLORS.accent};
            border-radius: {RADIUS.sm}px;
            padding: 7px 14px;
            font-weight: 600;
        """)
        bar.addWidget(self._tab_label)
        bar.addStretch()

        zoom_out = QPushButton("−")
        zoom_out.setFixedWidth(32)
        zoom_out.clicked.connect(self._zoom_out)
        bar.addWidget(zoom_out)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setStyleSheet(f"color: {COLORS.text_secondary}; background: transparent;")
        bar.addWidget(self._zoom_label)

        zoom_in = QPushButton("+")
        zoom_in.setFixedWidth(32)
        zoom_in.clicked.connect(self._zoom_in)
        bar.addWidget(zoom_in)

        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self._zoom_fit)
        bar.addWidget(fit_btn)

        for label, signal in [("OCR", self.ocr_requested), ("Translate", self.translate_requested), ("Inpaint", self.inpaint_requested)]:
            btn = QPushButton(label)
            btn.setProperty("accent", True)
            btn.clicked.connect(signal.emit)
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

        self._scene = QGraphicsScene()
        self._scene.setBackgroundBrush(QColor(COLORS.bg_base))
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self._view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._view.setMinimumSize(520, 360)
        self._view.setStyleSheet(f"border: 1px solid {COLORS.border}; border-radius: {RADIUS.md}px; background-color: {COLORS.bg_base};")
        lay.addWidget(self._view, 1)
        return wrapper

    def _build_thumbnail_strip(self) -> QWidget:
        panel = _Panel()
        panel.setFixedWidth(110)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)

        title = QLabel("Pages")
        title.setStyleSheet(f"color: {COLORS.text_primary}; font-weight: 700; background: transparent;")
        lay.addWidget(title)

        self._thumb_list = QListWidget()
        self._thumb_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._thumb_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._thumb_list.itemClicked.connect(self._on_thumbnail_clicked)
        lay.addWidget(self._thumb_list, 1)
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

        self._job_grid = QGridLayout()
        self._job_rows = [("OCR", "Idle", 0), ("Translate", "Idle", 0), ("Inpaint", "Idle", 0)]
        for r, (stage, status, pct) in enumerate(self._job_rows):
            self._job_grid.addWidget(QLabel(stage), r, 0)
            self._job_grid.addWidget(QLabel(status), r, 1)
            pb = QProgressBar()
            pb.setValue(pct)
            self._job_grid.addWidget(pb, r, 2)
        lay.addLayout(self._job_grid)
        return panel

    # ── Image loading & navigation ────────────────────────────
    def _load_current_image(self):
        self._scene.clear()
        self._pixmap_item = None
        path = self.current_image_path()
        if not path or not osp.isfile(path):
            self._scene.addText("No image loaded", QFont("Segoe UI", 16))
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._scene.addText(f"Failed to load\n{osp.basename(path)}", QFont("Segoe UI", 12))
            return
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._zoom_fit()

    def _refresh_thumbnails(self):
        self._thumb_list.clear()
        for i, path in enumerate(self._image_files):
            name = osp.basename(path)
            item = QListWidgetItem(f"{i+1:03d}: {name[:20]}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._thumb_list.addItem(item)
        if 0 <= self._current_index < self._thumb_list.count():
            self._thumb_list.setCurrentRow(self._current_index)

    def _on_thumbnail_clicked(self, item: QListWidgetItem):
        idx = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(idx, int) and 0 <= idx < len(self._image_files):
            self._current_index = idx
            self._load_current_image()
            self._update_tab_label()

    def _update_tab_label(self):
        path = self.current_image_path()
        if path:
            self._tab_label.setText(f"{osp.basename(path)}  ({self._current_index+1}/{len(self._image_files)})")
        else:
            self._tab_label.setText("No image open")

    # ── Zoom controls ───────────────────────────────────────
    def _zoom_in(self):
        self._view.scale(1.25, 1.25)
        self._update_zoom_label()

    def _zoom_out(self):
        self._view.scale(0.8, 0.8)
        self._update_zoom_label()

    def _zoom_fit(self):
        if self._pixmap_item:
            self._view.fitInView(self._pixmap_item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._update_zoom_label()

    def _update_zoom_label(self):
        scale = self._view.transform().m11()
        self._zoom_label.setText(f"{int(scale*100)}%")
