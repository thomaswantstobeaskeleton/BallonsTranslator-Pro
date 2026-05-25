from __future__ import annotations
from typing import List, Tuple, Optional, Callable
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QPlainTextEdit, QCheckBox, QSpinBox, QFormLayout,
    QMessageBox
)
from qtpy.QtCore import QTimer, Qt, QThread, QObject, Signal
import numpy as np

from utils.realtime_mode import (
    RealtimeRegion, RealtimeWatcher, RealtimeState,
    create_screenshot_backend, OverlayExclusionBackend,
    ScreenshotBackendBase
)
from utils.textblock import TextBlock
from .realtime_overlay import RealtimeOverlayWidget, TranslatedBlock


class _OcrTrWorker(QObject):
    """Runs OCR + translation off the UI thread."""
    finished = Signal(object)  # emits RealtimeState

    def __init__(self, img: np.ndarray, ocr_fn: Callable, tr_fn: Callable):
        super().__init__()
        self.img = img
        self.ocr_fn = ocr_fn
        self.tr_fn = tr_fn

    def run(self):
        state = RealtimeState()
        try:
            state.last_ocr_text = self.ocr_fn(self.img)
            if state.last_ocr_text:
                state.last_translation = self.tr_fn(state.last_ocr_text)
                state.status = "translated"
            else:
                state.status = "no_text"
        except Exception as e:
            state.status = "error"
            state.warnings.append(str(e))
        self.finished.emit(state)


class RealtimeTranslatorDialog(QDialog):
    """Realtime screen translator dialog with always-on-top overlay."""

    def __init__(self, parent=None, module_manager=None):
        super().__init__(parent)
        self._module_manager = module_manager
        self.setWindowTitle(self.tr("Realtime Screen Translator"))
        self.resize(520, 480)
        root = QVBoxLayout(self)

        # --- Module info row ---
        mod_row = QHBoxLayout()
        ocr_name = self._current_ocr_name()
        tr_name = self._current_translator_name()
        self.ocr_label = QLabel(self.tr("OCR: {}").format(ocr_name), self)
        self.tr_label = QLabel(self.tr("Translator: {}").format(tr_name), self)
        mod_row.addWidget(self.ocr_label)
        mod_row.addStretch()
        mod_row.addWidget(self.tr_label)
        root.addLayout(mod_row)

        self.privacy_hint = QLabel(
            self.tr("Privacy: captures/OCR/translation are not persisted or logged."), self)
        self.privacy_hint.setWordWrap(True)
        root.addWidget(self.privacy_hint)

        # --- Window selection ---
        win_row = QHBoxLayout()
        win_row.addWidget(QLabel(self.tr("Target window"), self))
        self.window_combo = QComboBox(self)
        self.window_combo.addItem(self.tr("Manual region"), "")
        self._refresh_window_list()
        win_row.addWidget(self.window_combo, stretch=1)
        self.refresh_win_btn = QPushButton(self.tr("Refresh"), self)
        self.refresh_win_btn.clicked.connect(self._refresh_window_list)
        win_row.addWidget(self.refresh_win_btn)
        root.addLayout(win_row)

        self.follow_window = QCheckBox(self.tr("Follow window position"), self)
        root.addWidget(self.follow_window)

        # --- Region + timing form ---
        form = QFormLayout()
        self.x_spin = QSpinBox(self); self.x_spin.setRange(0, 10000)
        self.y_spin = QSpinBox(self); self.y_spin.setRange(0, 10000)
        self.w_spin = QSpinBox(self); self.w_spin.setRange(1, 10000); self.w_spin.setValue(640)
        self.h_spin = QSpinBox(self); self.h_spin.setRange(1, 10000); self.h_spin.setValue(480)
        form.addRow(self.tr("Region X"), self.x_spin)
        form.addRow(self.tr("Region Y"), self.y_spin)
        form.addRow(self.tr("Region Width"), self.w_spin)
        form.addRow(self.tr("Region Height"), self.h_spin)

        self.capture_interval_ms = QSpinBox(self)
        self.capture_interval_ms.setRange(100, 5000)
        self.capture_interval_ms.setValue(700)
        self.min_ocr_interval_ms = QSpinBox(self)
        self.min_ocr_interval_ms.setRange(0, 5000)
        self.min_ocr_interval_ms.setValue(0)
        form.addRow(self.tr("Capture interval (ms)"), self.capture_interval_ms)
        form.addRow(self.tr("Min OCR interval (ms)"), self.min_ocr_interval_ms)
        root.addLayout(form)

        # --- Overlay controls ---
        overlay_row = QHBoxLayout()
        self.pin_overlay = QCheckBox(self.tr("Pin overlay (draggable)"), self)
        self.hide_overlay_btn = QPushButton(self.tr("Hide overlay"), self)
        self.show_overlay_btn = QPushButton(self.tr("Show overlay"), self)
        self.test_capture_btn = QPushButton(self.tr("Test capture"), self)
        overlay_row.addWidget(self.pin_overlay)
        overlay_row.addWidget(self.hide_overlay_btn)
        overlay_row.addWidget(self.show_overlay_btn)
        overlay_row.addWidget(self.test_capture_btn)
        root.addLayout(overlay_row)

        # --- Main buttons ---
        btns = QHBoxLayout()
        self.start_btn = QPushButton(self.tr("Start"), self)
        self.pause_btn = QPushButton(self.tr("Pause"), self)
        self.now_btn = QPushButton(self.tr("Translate now"), self)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.pause_btn)
        btns.addWidget(self.now_btn)
        root.addLayout(btns)

        # --- Output log ---
        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(200)
        root.addWidget(self.output)

        # --- Overlay widget ---
        self._overlay = RealtimeOverlayWidget(None)
        self._overlay.hide()

        # --- Backend & watcher ---
        self._fallback_backend = create_screenshot_backend("auto")
        self._wrapped_backend = OverlayExclusionBackend(
            self._fallback_backend,
            self._hide_overlay_for_capture,
            self._show_overlay_after_capture
        )
        self._watcher = RealtimeWatcher(
            self._wrapped_backend,
            self._ocr_image,
            self._translate_text
        )
        self._region = RealtimeRegion("default", (0, 0, 640, 480))
        self._pending_worker: Optional[QThread] = None

        # --- Timer ---
        self._timer = QTimer(self)
        self._timer.setInterval(self.capture_interval_ms.value())
        self._timer.timeout.connect(self._tick)
        self.capture_interval_ms.valueChanged.connect(self._timer.setInterval)
        self.min_ocr_interval_ms.valueChanged.connect(self._update_min_ocr_interval)

        # --- Connections ---
        self.start_btn.clicked.connect(self._on_start)
        self.pause_btn.clicked.connect(self._on_pause)
        self.now_btn.clicked.connect(self._tick)
        self.pin_overlay.toggled.connect(self._overlay.set_pinned)
        self.hide_overlay_btn.clicked.connect(self._overlay.hide)
        self.show_overlay_btn.clicked.connect(self._on_show_overlay)
        self.test_capture_btn.clicked.connect(self._on_test_capture)
        self.window_combo.currentIndexChanged.connect(self._on_window_changed)

        self._append_backend_info()

    # ------------------------------------------------------------------
    # Module helpers
    # ------------------------------------------------------------------
    def _current_ocr_name(self) -> str:
        if self._module_manager is None:
            return self.tr("not available")
        ocr = getattr(self._module_manager, "ocr", None)
        return getattr(ocr, "name", "unknown") if ocr else self.tr("not loaded")

    def _current_translator_name(self) -> str:
        if self._module_manager is None:
            return self.tr("not available")
        tr = getattr(self._module_manager, "translator", None)
        return getattr(tr, "name", "unknown") if tr else self.tr("not loaded")

    # ------------------------------------------------------------------
    # Window list
    # ------------------------------------------------------------------
    def _refresh_window_list(self):
        self.window_combo.clear()
        self.window_combo.addItem(self.tr("Manual region"), "")
        try:
            wins = self._fallback_backend.list_windows()
        except Exception:
            wins = []
        for w in wins:
            title = w.get("title", "") or ""
            wid = str(w.get("id", ""))
            if title and wid:
                display = f"{title[:50]} ({wid})"
                self.window_combo.addItem(display, wid)

    def _on_window_changed(self, idx: int):
        wid = self.window_combo.itemData(idx)
        if not wid:
            return
        # Try to auto-fill region from window rect
        try:
            rect = self._fallback_backend.resolve_follow_window_rect(wid)
            if rect:
                self.x_spin.setValue(rect[0])
                self.y_spin.setValue(rect[1])
                self.w_spin.setValue(rect[2])
                self.h_spin.setValue(rect[3])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Capture & overlay helpers
    # ------------------------------------------------------------------
    def _hide_overlay_for_capture(self):
        if self._overlay.isVisible():
            self._overlay.hide()

    def _show_overlay_after_capture(self):
        pass  # overlay is updated on next _tick finish

    def _on_show_overlay(self):
        x, y, w, h = self._current_region_rect()
        self._overlay.show_at(x, y, w, h)

    def _current_region_rect(self) -> Tuple[int, int, int, int]:
        return (self.x_spin.value(), self.y_spin.value(),
                self.w_spin.value(), self.h_spin.value())

    def _on_test_capture(self):
        x, y, w, h = self._current_region_rect()
        try:
            img = self._fallback_backend.capture_region((x, y, w, h))
            self.output.appendPlainText(
                self.tr("Test capture OK: shape={shape} dtype={dtype}").format(
                    shape=img.shape, dtype=img.dtype))
        except Exception as e:
            self.output.appendPlainText(self.tr("Test capture failed: {}").format(e))

    # ------------------------------------------------------------------
    # Start / pause
    # ------------------------------------------------------------------
    def _on_start(self):
        self._timer.start()
        self._overlay.show()

    def _on_pause(self):
        self._timer.stop()

    # ------------------------------------------------------------------
    # Tick: screenshot → background OCR/translate → update overlay
    # ------------------------------------------------------------------
    def _tick(self):
        if self._pending_worker is not None and self._pending_worker.isRunning():
            return  # skip if previous job still running

        # Build region
        x, y, w, h = self._current_region_rect()
        self._region.rect = (x, y, w, h)
        wid = self.window_combo.currentData()
        if wid and self.follow_window.isChecked():
            self._region.window_id = str(wid)
            self._region.follow_window = True
        else:
            self._region.window_id = ""
            self._region.follow_window = False

        # Capture image directly from backend
        capture_rect = (x, y, w, h)
        if wid and self.follow_window.isChecked() and self._wrapped_backend.supports_follow_window:
            resolved = self._wrapped_backend.resolve_follow_window_rect(str(wid))
            if resolved is not None:
                capture_rect = resolved
        try:
            img = self._wrapped_backend.capture_region(capture_rect)
        except Exception as e:
            self.output.appendPlainText(self.tr("Capture failed: {}").format(e))
            return
        if img is None or img.size == 0:
            return

        # Run OCR+translation in background thread
        worker = _OcrTrWorker(img, self._ocr_image, self._translate_text)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._pending_worker = thread
        thread.start()

    def _on_worker_finished(self, state: RealtimeState):
        self._pending_worker = None
        line = f"status={state.status} | src={state.last_ocr_text} | tr={state.last_translation}"
        self.output.appendPlainText(line)

        # Update overlay
        if state.last_translation and state.status == "translated":
            x, y, w, h = self._current_region_rect()
            blocks = [TranslatedBlock((0, 0, w, h), src_text=state.last_ocr_text, trans_text=state.last_translation)]
            self._overlay.set_blocks(blocks)
            self._overlay.show_at(x, y, w, h)
        else:
            self._overlay.set_blocks([])
            self._overlay.update()

    # ------------------------------------------------------------------
    # OCR & Translation — wired to ModuleManager pipeline
    # ------------------------------------------------------------------
    def _ocr_image(self, img: np.ndarray) -> str:
        """Run text detection + OCR on a captured image. Returns joined text."""
        if self._module_manager is None:
            return ""
        try:
            # 1. Detect text blocks
            detector = getattr(self._module_manager, "textdetector", None)
            if detector is None:
                return ""
            blk_list = detector.detect(img)
            if not blk_list:
                return ""
            # 2. Run OCR
            ocr = getattr(self._module_manager, "ocr", None)
            if ocr is None:
                return ""
            ocr.run_ocr(img, blk_list)
            parts = []
            for blk in blk_list:
                txt = getattr(blk, "get_text", lambda: "")()
                if txt and str(txt).strip():
                    parts.append(str(txt).strip())
            return "\n".join(parts)
        except Exception as e:
            return ""

    def _translate_text(self, text: str) -> str:
        """Translate a plain text string using the current translator."""
        if self._module_manager is None or not text:
            return ""
        try:
            translator = getattr(self._module_manager, "translator", None)
            if translator is None:
                return ""
            # Build a temporary TextBlock with the text
            blk = TextBlock()
            blk.text = [text]
            blk.translation = ""
            translator.translate_textblk_lst([blk])
            return str(getattr(blk, "translation", "") or "")
        except Exception:
            return ""

    def _update_min_ocr_interval(self):
        self._watcher.min_ocr_interval_sec = float(self.min_ocr_interval_ms.value()) / 1000.0

    def _append_backend_info(self):
        primary = getattr(self._wrapped_backend, "backend_name", "unknown")
        fallback = getattr(self._fallback_backend, "backend_name", "unknown")
        self.output.appendPlainText(f"capture_backend={primary} | fallback={fallback}")
        if not bool(getattr(self._wrapped_backend, "supports_window_exclusion", False)):
            self.output.appendPlainText("overlay_exclusion=fallback_hide_show")

    def closeEvent(self, event):
        self._timer.stop()
        self._overlay.hide()
        super().closeEvent(event)

