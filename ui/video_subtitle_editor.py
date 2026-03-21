"""
Video Subtitle Editor – CapCut-style editor for editing, cutting, and exporting video captions.

Features:
- Video preview with play/pause, seek, frame-accurate display
- Segment table: edit start/end times and text; add, delete, split at playhead
- Timeline strip: visual segments, click to select, drag handles to adjust in/out
- Trim/cut: in/out points, export or render only that range (segment times shifted)
- Undo/redo: history stack for table and timeline edits
- Per-segment styling: style (Default/Anime/Documentary) per caption
- Crop: crop region for render (left/top/right/bottom %)
- Export: SRT, ASS, WebVTT; render video with burned-in subtitles
"""
from __future__ import annotations

import os
import os.path as osp
import subprocess
from typing import List, Optional, Tuple, Any

from qtpy.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QSplitter,
    QGroupBox, QToolBar, QStatusBar, QFrame, QSpinBox, QDoubleSpinBox, QComboBox,
    QAbstractItemView, QMenu, QApplication, QProgressDialog, QScrollArea, QLineEdit,
)
from qtpy.QtCore import Qt, QTimer, Signal, QSize, QPoint
from qtpy.QtGui import QImage, QPixmap, QAction, QKeySequence, QPainter, QColor, QPen

from utils.logger import logger as LOGGER

# Segment: (start_sec, end_sec, text) or (start_sec, end_sec, text, style_override)
STYLE_OPTIONS = ["default", "anime", "documentary"]
MAX_UNDO = 50


def _sec_to_display(sec: float) -> str:
    """Format seconds as MM:SS.ms for display."""
    m = int(sec // 60)
    s = sec % 60
    return "%d:%05.2f" % (m, s)


def _frame_to_sec(frames: float, fps: float) -> float:
    return frames / fps if fps > 0 else 0


def _segment_tuple(s: float, e: float, text: str, style: Optional[str] = None) -> Tuple:
    if style:
        return (s, e, text, style)
    return (s, e, text)


class TimelineStrip(QWidget):
    """Horizontal strip showing subtitle segments; click to select, drag handles to adjust (frame-accurate with fps)."""
    segmentSelected = Signal(int)  # index
    segmentTimesChanged = Signal(int, float, float)  # index, start_sec, end_sec

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(48)
        self.setMaximumHeight(72)
        self._segments: List[Tuple] = []  # (start_sec, end_sec, text) or (..., style)
        self._duration_sec = 1.0
        self._fps = 25.0
        self._selected_index = -1
        self._hover_index = -1
        self._drag_mode = None  # "start" | "end" | None
        self._drag_index = -1
        self.setMouseTracking(True)

    def set_duration(self, duration_sec: float):
        self._duration_sec = max(0.01, duration_sec)
        self.update()

    def set_fps(self, fps: float):
        self._fps = max(1.0, fps)
        self.update()

    def set_segments(self, segments: List[Tuple[float, float, str]]):
        self._segments = list(segments)
        self.update()

    def set_selected_index(self, index: int):
        self._selected_index = index
        self.update()

    def _x_to_sec(self, x: int) -> float:
        w = self.width()
        if w <= 0:
            return 0.0
        return (x / w) * self._duration_sec

    def _sec_to_x(self, sec: float) -> int:
        w = self.width()
        if self._duration_sec <= 0:
            return 0
        return int((sec / self._duration_sec) * w)

    def _hit_segment(self, x: int) -> Tuple[int, Optional[str]]:
        """Return (segment_index, 'start'|'end'|'body'|None)."""
        for i, seg in enumerate(self._segments):
            s, e = seg[0], seg[1]
            xs, xe = self._sec_to_x(s), self._sec_to_x(e)
            handle = 8
            if abs(x - xs) <= handle:
                return (i, "start")
            if abs(x - xe) <= handle:
                return (i, "end")
            if xs <= x <= xe:
                return (i, "body")
        return (-1, None)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x() if hasattr(event, "position") else event.x()
        idx, mode = self._hit_segment(int(x))
        if idx >= 0:
            self._selected_index = idx
            self.segmentSelected.emit(idx)
        self._drag_mode = mode
        self._drag_index = idx
        self.update()

    def mouseMoveEvent(self, event):
        x = int(event.position().x() if hasattr(event, "position") else event.x())
        if self._drag_mode and self._drag_index >= 0 and self._drag_index < len(self._segments):
            seg = self._segments[self._drag_index]
            s, e, text = seg[0], seg[1], seg[2]
            style = seg[3] if len(seg) > 3 else None
            sec = self._x_to_sec(x)
            sec = max(0.0, min(self._duration_sec, sec))
            if self._drag_mode == "start":
                s = min(sec, e - 0.1)
                self._segments[self._drag_index] = _segment_tuple(s, e, text, style)
                self.segmentTimesChanged.emit(self._drag_index, s, e)
            elif self._drag_mode == "end":
                e = max(sec, s + 0.1)
                self._segments[self._drag_index] = _segment_tuple(s, e, text, style)
                self.segmentTimesChanged.emit(self._drag_index, s, e)
        else:
            idx, _ = self._hit_segment(x)
            self._hover_index = idx
        self.update()

    def mouseReleaseEvent(self, event):
        self._drag_mode = None
        self._drag_index = -1
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0 or self._duration_sec <= 0:
            return
        # Background
        painter.fillRect(0, 0, w, h, QColor(40, 40, 40))
        for i, seg in enumerate(self._segments):
            s, e = seg[0], seg[1]
            xs, xe = self._sec_to_x(s), self._sec_to_x(e)
            if xe <= xs:
                continue
            is_sel = i == self._selected_index
            is_hov = i == self._hover_index
            if is_sel:
                painter.fillRect(xs, 2, xe - xs, h - 4, QColor(70, 130, 180))
            elif is_hov:
                painter.fillRect(xs, 2, xe - xs, h - 4, QColor(60, 60, 80))
            else:
                painter.fillRect(xs, 4, xe - xs, h - 8, QColor(80, 80, 100))
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.drawRect(xs, 2, xe - xs, h - 4)
        painter.end()


class VideoSubtitleEditorWindow(QMainWindow):
    """Main window for the Video Subtitle Editor: preview, segment table, timeline, export."""

    def __init__(self, parent=None, video_path: str = "", subtitle_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Video Subtitle Editor"))
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)

        # Video state
        self._video_path = ""
        self._cap = None
        self._fps = 25.0
        self._total_frames = 0
        self._duration_sec = 0.0
        self._current_frame = 0
        self._playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        # Segments: (start_sec, end_sec, text) or (start_sec, end_sec, text, style)
        self._segments: List[Tuple] = []
        self._in_sec: Optional[float] = None
        self._out_sec: Optional[float] = None
        self._undo_stack: List[List[Tuple]] = []
        self._redo_stack: List[List[Tuple]] = []

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Toolbar
        toolbar = QToolBar(self.tr("File"))
        toolbar.setMovable(False)
        open_video_act = QAction(self.tr("Open video..."), self)
        open_video_act.triggered.connect(self._open_video)
        toolbar.addAction(open_video_act)
        load_srt_act = QAction(self.tr("Load subtitles..."), self)
        load_srt_act.triggered.connect(self._load_subtitles)
        toolbar.addAction(load_srt_act)
        toolbar.addSeparator()
        save_srt_act = QAction(self.tr("Save SRT..."), self)
        save_srt_act.triggered.connect(self._save_srt)
        toolbar.addAction(save_srt_act)
        export_ass_act = QAction(self.tr("Export ASS..."), self)
        export_ass_act.triggered.connect(self._export_ass)
        toolbar.addAction(export_ass_act)
        export_vtt_act = QAction(self.tr("Export WebVTT..."), self)
        export_vtt_act.triggered.connect(self._export_vtt)
        toolbar.addAction(export_vtt_act)
        toolbar.addSeparator()
        render_act = QAction(self.tr("Render video (burn subtitles)..."), self)
        render_act.triggered.connect(self._render_video)
        toolbar.addAction(render_act)
        toolbar.addSeparator()
        self.undo_act = QAction(self.tr("Undo"), self)
        self.undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_act.triggered.connect(self._undo)
        toolbar.addAction(self.undo_act)
        self.redo_act = QAction(self.tr("Redo"), self)
        self.redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_act.triggered.connect(self._redo)
        toolbar.addAction(self.redo_act)
        self.undo_act.setEnabled(False)
        self.redo_act.setEnabled(False)
        self.addToolBar(toolbar)

        # Main splitter: preview left, table + timeline right
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: video preview + playback
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(480, 270)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #1a1a1a; color: #888;")
        self.preview_label.setText(self.tr("Open a video to preview"))
        left_layout.addWidget(self.preview_label)
        ctrl = QHBoxLayout()
        self.play_btn = QPushButton(self.tr("Play"))
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        ctrl.addWidget(self.play_btn)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderMoved.connect(self._seek_to)
        self.seek_slider.sliderPressed.connect(lambda: setattr(self, "_seek_pressed", True))
        self.seek_slider.sliderReleased.connect(lambda: setattr(self, "_seek_pressed", False))
        ctrl.addWidget(self.seek_slider, 1)
        self.time_label = QLabel("0:00.00 / 0:00.00")
        ctrl.addWidget(self.time_label)
        left_layout.addLayout(ctrl)
        splitter.addWidget(left)

        # Right: segment table + timeline
        right = QWidget()
        right_layout = QVBoxLayout(right)
        # Timeline
        self.timeline = TimelineStrip(self)
        self.timeline.segmentSelected.connect(self._on_timeline_segment_selected)
        self.timeline.segmentTimesChanged.connect(self._on_timeline_times_changed)
        right_layout.addWidget(QLabel(self.tr("Timeline")))
        right_layout.addWidget(self.timeline)

        # Table
        tbl_group = QGroupBox(self.tr("Subtitles"))
        tbl_layout = QVBoxLayout(tbl_group)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([self.tr("Start"), self.tr("End"), self.tr("Text"), self.tr("Style")])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.cellChanged.connect(self._on_table_cell_changed)
        tbl_layout.addWidget(self.table)
        btn_row = QHBoxLayout()
        add_btn = QPushButton(self.tr("Add segment"))
        add_btn.clicked.connect(self._add_segment)
        delete_btn = QPushButton(self.tr("Delete"))
        delete_btn.clicked.connect(self._delete_segment)
        split_btn = QPushButton(self.tr("Split at playhead"))
        split_btn.clicked.connect(self._split_at_playhead)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(delete_btn)
        btn_row.addWidget(split_btn)
        btn_row.addStretch()
        tbl_layout.addLayout(btn_row)
        right_layout.addWidget(tbl_group)
        splitter.addWidget(right)

        # Trim (in/out) and crop
        trim_group = QGroupBox(self.tr("Trim & crop"))
        trim_l = QVBoxLayout(trim_group)
        inout_row = QHBoxLayout()
        inout_row.addWidget(QLabel(self.tr("In:")))
        self.in_edit = QLineEdit()
        self.in_edit.setPlaceholderText(self.tr("0:00 (leave empty = start)"))
        self.in_edit.setMaximumWidth(80)
        inout_row.addWidget(self.in_edit)
        self.set_in_btn = QPushButton(self.tr("Set at playhead"))
        self.set_in_btn.clicked.connect(self._set_in_at_playhead)
        inout_row.addWidget(self.set_in_btn)
        inout_row.addWidget(QLabel(self.tr("Out:")))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText(self.tr("end (leave empty = end)"))
        self.out_edit.setMaximumWidth(80)
        inout_row.addWidget(self.out_edit)
        self.set_out_btn = QPushButton(self.tr("Set at playhead"))
        self.set_out_btn.clicked.connect(self._set_out_at_playhead)
        inout_row.addWidget(self.set_out_btn)
        clear_inout_btn = QPushButton(self.tr("Clear In/Out"))
        clear_inout_btn.clicked.connect(self._clear_in_out)
        inout_row.addWidget(clear_inout_btn)
        inout_row.addStretch()
        trim_l.addLayout(inout_row)
        crop_row = QHBoxLayout()
        crop_row.addWidget(QLabel(self.tr("Crop % (L,T,R,B):")))
        self.crop_l = QSpinBox()
        self.crop_l.setRange(0, 49)
        self.crop_l.setValue(0)
        self.crop_l.setSuffix("%")
        self.crop_t = QSpinBox()
        self.crop_t.setRange(0, 49)
        self.crop_t.setValue(0)
        self.crop_t.setSuffix("%")
        self.crop_r = QSpinBox()
        self.crop_r.setRange(0, 49)
        self.crop_r.setValue(0)
        self.crop_r.setSuffix("%")
        self.crop_b = QSpinBox()
        self.crop_b.setRange(0, 49)
        self.crop_b.setValue(0)
        self.crop_b.setSuffix("%")
        crop_row.addWidget(self.crop_l)
        crop_row.addWidget(self.crop_t)
        crop_row.addWidget(self.crop_r)
        crop_row.addWidget(self.crop_b)
        crop_row.addStretch()
        trim_l.addLayout(crop_row)
        layout.addWidget(trim_group)

        splitter.setSizes([500, 400])
        layout.addWidget(splitter)

        # Render options
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel(self.tr("Subtitle style:")))
        self.style_combo = QComboBox()
        self.style_combo.addItems([self.tr("Default"), self.tr("Anime (larger)"), self.tr("Documentary (smaller)")])
        opt_row.addWidget(self.style_combo)
        opt_row.addStretch()
        layout.addLayout(opt_row)

        self.statusBar().showMessage(self.tr("Open a video and optionally load subtitles (SRT/ASS/VTT)."))

        if video_path and osp.isfile(video_path):
            self._video_path = video_path
            self._open_video_inner()
        if subtitle_path and osp.isfile(subtitle_path):
            self._load_subtitles_inner(subtitle_path)

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Open video"), "",
            self.tr("Video files (*.mp4 *.avi *.mkv *.mov *.webm);;All files (*)")
        )
        if not path:
            return
        self._video_path = path
        self._open_video_inner()

    def _open_video_inner(self):
        try:
            import cv2
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._cap = cv2.VideoCapture(self._video_path)
            if not self._cap.isOpened():
                QMessageBox.warning(self, self.tr("Error"), self.tr("Could not open video: %s") % self._video_path)
                return
            self._fps = max(0.001, self._cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self._duration_sec = self._total_frames / self._fps if self._fps > 0 else 0
            self._current_frame = 0
            self._playing = False
            self._play_timer.stop()
            self.seek_slider.setRange(0, max(0, self._total_frames - 1))
            self.seek_slider.setValue(0)
            self.play_btn.setEnabled(True)
            self.timeline.set_duration(self._duration_sec)
            self.timeline.set_fps(self._fps)
            self.timeline.set_segments(self._segments)
            self._update_time_label()
            self._show_frame_at(self._current_frame)
            self.statusBar().showMessage(self.tr("Loaded: %s") % osp.basename(self._video_path))
        except Exception as e:
            LOGGER.exception("Open video")
            QMessageBox.warning(self, self.tr("Error"), self.tr("Failed to open video: %s") % str(e))

    def _load_subtitles(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Load subtitles"), osp.dirname(self._video_path) if self._video_path else "",
            self.tr("Subtitle files (*.srt *.ass *.vtt);;All files (*)")
        )
        if not path:
            return
        self._load_subtitles_inner(path)

    def _load_subtitles_inner(self, path: str):
        from modules.video_subtitle_extract import parse_srt, parse_vtt, parse_ass
        ext = osp.splitext(path)[1].lower()
        try:
            if ext == ".srt":
                segs = parse_srt(path)
            elif ext == ".vtt":
                segs = parse_vtt(path)
            elif ext == ".ass":
                segs = parse_ass(path)
            else:
                segs = parse_srt(path)
        except Exception as e:
            LOGGER.exception("Load subtitles")
            QMessageBox.warning(self, self.tr("Error"), self.tr("Failed to load subtitles: %s") % str(e))
            return
        self._segments = [_segment_tuple(s, e, t, None) for s, e, t in segs]
        self._rebuild_table()
        self.timeline.set_segments(self._segments)
        self.statusBar().showMessage(self.tr("Loaded %d segments from %s") % (len(segs), osp.basename(path)))

    def _rebuild_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._segments))
        style_labels = [self.tr("Default"), self.tr("Anime"), self.tr("Documentary")]
        for i, seg in enumerate(self._segments):
            s, e = seg[0], seg[1]
            text = seg[2] if len(seg) > 2 else ""
            style = (seg[3] if len(seg) > 3 and seg[3] else None) or "default"
            self.table.setItem(i, 0, QTableWidgetItem(_sec_to_display(s)))
            self.table.setItem(i, 1, QTableWidgetItem(_sec_to_display(e)))
            self.table.setItem(i, 2, QTableWidgetItem(text or ""))
            style_combo = QComboBox()
            style_combo.addItems(style_labels)
            style_combo.setCurrentIndex(STYLE_OPTIONS.index(style) if style in STYLE_OPTIONS else 0)
            style_combo.currentIndexChanged.connect(lambda idx, r=i: self._on_style_changed(r, idx))
            self.table.setCellWidget(i, 3, style_combo)
        self.table.blockSignals(False)

    def _segments_from_table(self) -> List[Tuple]:
        def parse_display(t: str) -> float:
            t = (t or "").strip()
            if ":" in t:
                parts = t.split(":")
                if len(parts) >= 2:
                    try:
                        m = int(parts[0])
                        s = float(parts[1].replace(",", "."))
                        return m * 60 + s
                    except ValueError:
                        pass
            try:
                return float(t.replace(",", "."))
            except ValueError:
                return 0.0
        out = []
        for r in range(self.table.rowCount()):
            start_s = parse_display(self.table.item(r, 0).text() if self.table.item(r, 0) else "")
            end_s = parse_display(self.table.item(r, 1).text() if self.table.item(r, 1) else "")
            text = (self.table.item(r, 2).text() if self.table.item(r, 2) else "").strip()
            style = None
            w = self.table.cellWidget(r, 3)
            if isinstance(w, QComboBox):
                idx = w.currentIndex()
                if 0 <= idx < len(STYLE_OPTIONS):
                    style = STYLE_OPTIONS[idx]
            out.append(_segment_tuple(start_s, end_s, text, style))
        return out

    def _on_style_changed(self, row: int, index: int):
        if row < 0 or row >= len(self._segments):
            return
        style = STYLE_OPTIONS[index] if 0 <= index < len(STYLE_OPTIONS) else None
        seg = self._segments[row]
        self._segments[row] = _segment_tuple(seg[0], seg[1], seg[2], style)
        self.timeline.set_segments(self._segments)

    def _push_undo(self):
        state = [tuple(seg) for seg in self._segments]
        self._undo_stack.append(state)
        if len(self._undo_stack) > MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self.undo_act.setEnabled(True)
        self.redo_act.setEnabled(False)

    def _undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append([tuple(seg) for seg in self._segments])
        self._segments = list(self._undo_stack.pop())
        self._rebuild_table()
        self.timeline.set_segments(self._segments)
        self.undo_act.setEnabled(len(self._undo_stack) > 0)
        self.redo_act.setEnabled(True)

    def _redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append([tuple(seg) for seg in self._segments])
        self._segments = list(self._redo_stack.pop())
        self._rebuild_table()
        self.timeline.set_segments(self._segments)
        self.undo_act.setEnabled(True)
        self.redo_act.setEnabled(len(self._redo_stack) > 0)

    def _parse_in_out(self) -> Tuple[Optional[float], Optional[float]]:
        def parse(t: str) -> Optional[float]:
            t = (t or "").strip()
            if not t:
                return None
            if ":" in t:
                parts = t.split(":")
                if len(parts) >= 2:
                    try:
                        m = int(parts[0])
                        s = float(parts[1].replace(",", "."))
                        return m * 60 + s
                    except ValueError:
                        pass
            try:
                return float(t.replace(",", "."))
            except ValueError:
                return None
        return parse(self.in_edit.text()), parse(self.out_edit.text())

    def _set_in_at_playhead(self):
        t = self._current_frame / self._fps if self._fps > 0 else 0
        self.in_edit.setText(_sec_to_display(t))
        self._in_sec = t

    def _set_out_at_playhead(self):
        t = self._current_frame / self._fps if self._fps > 0 else 0
        self.out_edit.setText(_sec_to_display(t))
        self._out_sec = t

    def _clear_in_out(self):
        self.in_edit.clear()
        self.out_edit.clear()
        self._in_sec = self._out_sec = None

    def _get_trimmed_segments(self) -> List[Tuple]:
        """Segments in [in_sec, out_sec], clipped and shifted so start is 0."""
        in_s, out_s = self._parse_in_out()
        if in_s is None:
            in_s = 0.0
        if out_s is None:
            out_s = self._duration_sec
        in_s = max(0, min(in_s, self._duration_sec))
        out_s = max(in_s + 0.01, min(out_s, self._duration_sec))
        result = []
        for seg in self._segments:
            s, e = seg[0], seg[1]
            text = seg[2] if len(seg) > 2 else ""
            style = seg[3] if len(seg) > 3 else None
            if e <= in_s or s >= out_s:
                continue
            ns = max(s, in_s) - in_s
            ne = min(e, out_s) - in_s
            if ne > ns:
                result.append(_segment_tuple(ns, ne, text, style))
        return result

    def _sync_segments_from_table(self):
        self._segments = self._segments_from_table()
        self.timeline.set_segments(self._segments)

    def _on_table_selection_changed(self):
        rows = self.table.selectedIndexes()
        if not rows:
            self.timeline.set_selected_index(-1)
            return
        self.timeline.set_selected_index(rows[0].row())

    def _on_table_cell_changed(self, row: int, col: int):
        self._sync_segments_from_table()
        self.timeline.set_segments(self._segments)

    def _on_timeline_segment_selected(self, index: int):
        self.table.selectRow(index)
        self.table.setFocus()

    def _on_timeline_times_changed(self, index: int, start_sec: float, end_sec: float):
        if index < 0 or index >= len(self._segments):
            return
        seg = self._segments[index]
        style = seg[3] if len(seg) > 3 else None
        self._segments[index] = _segment_tuple(start_sec, end_sec, seg[2], style)
        self.table.blockSignals(True)
        self.table.item(index, 0).setText(_sec_to_display(start_sec))
        self.table.item(index, 1).setText(_sec_to_display(end_sec))
        self.table.blockSignals(False)

    def _add_segment(self):
        self._push_undo()
        t = self._current_frame / self._fps if self._fps > 0 else 0
        self._segments.append(_segment_tuple(t, t + 5.0, "", None))
        self._segments.sort(key=lambda x: x[0])
        self._rebuild_table()
        self.timeline.set_segments(self._segments)

    def _delete_segment(self):
        row = self.table.currentRow()
        if row < 0:
            return
        self._push_undo()
        self._segments.pop(row)
        self._rebuild_table()
        self.timeline.set_segments(self._segments)

    def _split_at_playhead(self):
        t = self._current_frame / self._fps if self._fps > 0 else 0
        for i, seg in enumerate(self._segments):
            s, e, text = seg[0], seg[1], seg[2]
            style = seg[3] if len(seg) > 3 else None
            if s <= t < e and (e - t) > 0.1 and (t - s) > 0.1:
                self._push_undo()
                self._segments[i] = _segment_tuple(s, t, text, style)
                self._segments.insert(i + 1, _segment_tuple(t, e, text, style))
                self._rebuild_table()
                self.timeline.set_segments(self._segments)
                self.statusBar().showMessage(self.tr("Split segment at %s") % _sec_to_display(t))
                return
        self.statusBar().showMessage(self.tr("Playhead not inside a segment or segment too short to split."))

    def _toggle_play(self):
        self._playing = not self._playing
        self.play_btn.setText(self.tr("Pause") if self._playing else self.tr("Play"))
        if self._playing:
            self._play_timer.start(max(1, int(1000 / self._fps)))
        else:
            self._play_timer.stop()

    def _on_play_tick(self):
        if self._cap is None or not self._cap.isOpened():
            self._play_timer.stop()
            return
        self._current_frame += 1
        if self._total_frames > 0 and self._current_frame >= self._total_frames:
            self._current_frame = self._total_frames - 1
            self._playing = False
            self.play_btn.setText(self.tr("Play"))
            self._play_timer.stop()
        if not getattr(self, "_seek_pressed", False):
            self.seek_slider.setValue(self._current_frame)
        self._update_time_label()
        self._show_frame_at(self._current_frame)

    def _seek_to(self, frame: int):
        self._current_frame = max(0, min(frame, self._total_frames - 1 if self._total_frames > 0 else 0))
        if self._cap is not None and self._cap.isOpened():
            import cv2
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)
        self._update_time_label()
        self._show_frame_at(self._current_frame)

    def _update_time_label(self):
        cur_sec = _frame_to_sec(self._current_frame, self._fps)
        dur_sec = self._duration_sec
        self.time_label.setText("%s / %s" % (_sec_to_display(cur_sec), _sec_to_display(dur_sec)))

    def _show_frame_at(self, frame: int):
        if self._cap is None or not self._cap.isOpened():
            return
        try:
            import cv2
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame_bgr = self._cap.read()
            if not ret or frame_bgr is None:
                return
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            bytes_per_line = rgb.strides[0]
            img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(img).scaled(
                self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except Exception as e:
            LOGGER.debug("Preview frame: %s", e)

    def _save_srt(self):
        if not self._video_path or not self._cap:
            QMessageBox.warning(self, self.tr("Error"), self.tr("Open a video first."))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Save SRT"), osp.splitext(self._video_path)[0] + ".srt",
            self.tr("SRT files (*.srt);;All files (*)")
        )
        if not path:
            return
        self._sync_segments_from_table()
        self._write_srt_file(path)
        self.statusBar().showMessage(self.tr("Saved SRT: %s") % path)

    def _write_srt_file(self, path: str):
        from .video_translator_dialog import _write_srt
        trimmed = self._get_trimmed_segments()
        entries = [(int(seg[0] * self._fps), int(seg[1] * self._fps), seg[2]) for seg in trimmed]
        _write_srt(path, entries, self._fps)

    def _export_ass(self):
        if not self._video_path or not self._cap:
            QMessageBox.warning(self, self.tr("Error"), self.tr("Open a video first."))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export ASS"), osp.splitext(self._video_path)[0] + ".ass",
            self.tr("ASS files (*.ass);;All files (*)")
        )
        if not path:
            return
        self._sync_segments_from_table()
        from .video_translator_dialog import _write_ass
        trimmed = self._get_trimmed_segments()
        entries = [(int(seg[0] * self._fps), int(seg[1] * self._fps), seg[2]) for seg in trimmed]
        _write_ass(path, entries, self._fps)
        self.statusBar().showMessage(self.tr("Exported ASS: %s") % path)

    def _export_vtt(self):
        if not self._video_path or not self._cap:
            QMessageBox.warning(self, self.tr("Error"), self.tr("Open a video first."))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export WebVTT"), osp.splitext(self._video_path)[0] + ".vtt",
            self.tr("WebVTT files (*.vtt);;All files (*)")
        )
        if not path:
            return
        self._sync_segments_from_table()
        from .video_translator_dialog import _write_vtt
        trimmed = self._get_trimmed_segments()
        entries = [(int(seg[0] * self._fps), int(seg[1] * self._fps), seg[2]) for seg in trimmed]
        _write_vtt(path, entries, self._fps)
        self.statusBar().showMessage(self.tr("Exported WebVTT: %s") % path)

    def _render_video(self):
        if not self._video_path or self._cap is None:
            QMessageBox.warning(self, self.tr("Error"), self.tr("Open a video first."))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Render video"), osp.splitext(self._video_path)[0] + "_subtitled.mp4",
            self.tr("Video files (*.mp4 *.avi);;All files (*)")
        )
        if not path:
            return
        self._sync_segments_from_table()
        in_s, out_s = self._parse_in_out()
        in_frame = int((in_s or 0) * self._fps) if self._fps > 0 else 0
        out_frame = int((out_s or self._duration_sec) * self._fps) if self._fps > 0 else self._total_frames
        out_frame = min(out_frame, self._total_frames)
        trimmed = self._get_trimmed_segments()
        # Build segments for draw: (s, e, text) or (s, e, text, style)
        segments_for_draw = [tuple(seg) for seg in trimmed]
        cl, ct, cr, cb = self.crop_l.value(), self.crop_t.value(), self.crop_r.value(), self.crop_b.value()
        try:
            import cv2
            from modules.video_translator import _draw_timed_subs_on_image, subtitle_black_box_draw_kwargs_from_cfg
            from utils.config import pcfg as _pcfg_sub
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                QMessageBox.warning(self, self.tr("Error"), self.tr("Could not open video for rendering."))
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            # Crop region (percent to remove from each side)
            x1 = int(w * cl / 100)
            y1 = int(h * ct / 100)
            x2 = int(w * (100 - cr) / 100)
            y2 = int(h * (100 - cb) / 100)
            out_w = max(1, x2 - x1)
            out_h = max(1, y2 - y1)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(path, fourcc, fps, (out_w, out_h))
            if not out.isOpened():
                out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), fps, (out_w, out_h))
            if not out.isOpened():
                cap.release()
                QMessageBox.warning(self, self.tr("Error"), self.tr("Could not create output video file."))
                return
            prog = QProgressDialog(self.tr("Rendering video..."), self.tr("Cancel"), 0, max(1, out_frame - in_frame), self)
            prog.setWindowModality(Qt.WindowModality.WindowModal)
            cap.set(cv2.CAP_PROP_POS_FRAMES, in_frame)
            n = in_frame
            step = max(1, (out_frame - in_frame) // 100)
            while n < out_frame:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                time_sec = (n - in_frame) / fps if fps > 0 else 0
                if cl or ct or cr or cb:
                    frame = frame[y1:y2, x1:x2]
                _draw_timed_subs_on_image(
                    frame,
                    time_sec,
                    segments_for_draw,
                    style=style,
                    **subtitle_black_box_draw_kwargs_from_cfg(_pcfg_sub.module),
                )
                out.write(frame)
                n += 1
                if (n - in_frame) % step == 0:
                    prog.setValue(n - in_frame)
                    QApplication.processEvents()
                    if prog.wasCanceled():
                        break
            cap.release()
            out.release()
            prog.close()
            self.statusBar().showMessage(self.tr("Rendered: %s") % path)
            QMessageBox.information(self, self.tr("Done"), self.tr("Video saved to: %s") % path)
        except Exception as e:
            LOGGER.exception("Render video")
            QMessageBox.warning(self, self.tr("Error"), self.tr("Render failed: %s") % str(e))

    def closeEvent(self, event):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        event.accept()
