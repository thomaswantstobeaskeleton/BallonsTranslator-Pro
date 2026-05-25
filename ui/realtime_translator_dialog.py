from __future__ import annotations
from typing import List, Tuple, Optional, Callable
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QPlainTextEdit, QCheckBox, QSpinBox, QFormLayout,
    QMessageBox, QApplication, QWidget, QTextBrowser
)
from qtpy.QtCore import QTimer, Qt, QThread, Signal, QPoint, QRect
from qtpy.QtGui import QPainter, QPen, QBrush, QColor, QBitmap, QFont, QTextCharFormat, QImage
import numpy as np

try:
    from cv2 import cvtColor, COLOR_BGR2GRAY
    from skimage.metrics import structural_similarity as ssim
    _CV2_OK = True
except Exception:
    _CV2_OK = False

from utils.realtime_mode import (
    RealtimeRegion, RealtimeWatcher, RealtimeState,
    create_screenshot_backend, OverlayExclusionBackend,
    ScreenshotBackendBase
)
from utils.textblock import TextBlock
from .realtime_overlay import RealtimeOverlayWidget, TranslatedBlock


class ScreenRangeSelector(QWidget):
    """Fullscreen overlay for drag-to-select a screen region (Dango-style)."""

    region_selected = Signal(tuple)  # (x, y, w, h)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setStyleSheet("background-color:black;")
        self.setWindowOpacity(0.4)

        screens = QApplication.screens()
        max_width, max_height = 0, 0
        for sc in screens:
            geom = sc.geometry()
            max_height = max(max_height, geom.height())
            max_width += geom.width()
        self.setGeometry(0, 0, max_width - 1, max_height - 1)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._is_drawing = False
        self._start_point = QPoint()
        self._end_point = QPoint()

    def paintEvent(self, event):
        if not self._is_drawing:
            return
        painter = QPainter(self)
        pen = QPen(QColor(0, 120, 255), 2)
        painter.setPen(pen)
        brush = QBrush(QColor(0, 120, 255, 60))
        painter.setBrush(brush)
        rect = QRect(self._start_point, self._end_point).normalized()
        painter.drawRect(rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_point = event.pos()
            self._end_point = self._start_point
            self._is_drawing = True

    def mouseMoveEvent(self, event):
        if self._is_drawing:
            self._end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._is_drawing:
            self._is_drawing = False
            self._end_point = event.pos()
            rect = QRect(self._start_point, self._end_point).normalized()
            self.region_selected.emit((rect.x(), rect.y(), rect.width(), rect.height()))
            self.close()


class _OcrTrWorker(QThread):
    """Runs OCR + translation off the UI thread (Dango-style QThread subclass)."""
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
    """Realtime screen translator dialog with Dango-style range selection and result window."""

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

        # --- Region selection ---
        region_row = QHBoxLayout()
        self.select_range_btn = QPushButton(self.tr("Select Screen Region"), self)
        self.select_range_btn.setToolTip(self.tr("Drag to select the screen area to capture and translate."))
        self.auto_mode_cb = QCheckBox(self.tr("Auto-translate mode"), self)
        self.auto_mode_cb.setToolTip(self.tr("Continuously capture and translate the selected region."))
        region_row.addWidget(self.select_range_btn)
        region_row.addWidget(self.auto_mode_cb)
        root.addLayout(region_row)

        # --- Region display ---
        self.region_label = QLabel(self.tr("Region: not set"), self)
        root.addWidget(self.region_label)

        # --- Timing form ---
        form = QFormLayout()
        self.capture_interval_ms = QSpinBox(self)
        self.capture_interval_ms.setRange(100, 5000)
        self.capture_interval_ms.setValue(700)
        self.min_ocr_interval_ms = QSpinBox(self)
        self.min_ocr_interval_ms.setRange(0, 5000)
        self.min_ocr_interval_ms.setValue(0)
        self.ssim_threshold = QSpinBox(self)
        self.ssim_threshold.setRange(50, 100)
        self.ssim_threshold.setValue(95)
        self.ssim_threshold.setSuffix("%")
        form.addRow(self.tr("Capture interval (ms)"), self.capture_interval_ms)
        form.addRow(self.tr("Min OCR interval (ms)"), self.min_ocr_interval_ms)
        form.addRow(self.tr("Image similarity threshold"), self.ssim_threshold)
        root.addLayout(form)

        # --- Main buttons ---
        btns = QHBoxLayout()
        self.start_btn = QPushButton(self.tr("Start"), self)
        self.pause_btn = QPushButton(self.tr("Pause"), self)
        self.now_btn = QPushButton(self.tr("Translate now"), self)
        self.test_capture_btn = QPushButton(self.tr("Test Capture"), self)
        self.hide_result_btn = QPushButton(self.tr("Hide Result"), self)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.pause_btn)
        btns.addWidget(self.now_btn)
        btns.addWidget(self.test_capture_btn)
        btns.addWidget(self.hide_result_btn)
        root.addLayout(btns)

        # --- Output log ---
        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(200)
        root.addWidget(self.output)

        # --- Result window (frameless always-on-top) ---
        self._result_window = _TranslationResultWindow()
        self._result_window.hide()

        # --- Legacy overlay (kept for compatibility) ---
        self._overlay = RealtimeOverlayWidget(None)
        self._overlay.hide()

        # --- State ---
        self._region: Optional[Tuple[int, int, int, int]] = None
        self._last_image: Optional[np.ndarray] = None
        self._last_blocks: List = []  # Store detected blocks for positioning
        self._pending_worker: Optional[QThread] = None
        self._stop_sign = False

        # --- Timer ---
        self._timer = QTimer(self)
        self._timer.setInterval(self.capture_interval_ms.value())
        self._timer.timeout.connect(self._tick)
        self.capture_interval_ms.valueChanged.connect(self._timer.setInterval)
        self.min_ocr_interval_ms.valueChanged.connect(self._update_min_ocr_interval)

        # --- Module readiness poll ---
        self._ready_timer = QTimer(self)
        self._ready_timer.setInterval(500)
        self._ready_timer.timeout.connect(self._update_start_button_state)
        self._ready_timer.start()

        # --- Connections ---
        self.start_btn.clicked.connect(self._on_start)
        self.pause_btn.clicked.connect(self._on_pause)
        self.now_btn.clicked.connect(self._on_translate_now)
        self.test_capture_btn.clicked.connect(self._on_test_capture)
        self.hide_result_btn.clicked.connect(self._result_window.hide)
        self.select_range_btn.clicked.connect(self._on_select_range)
        self.auto_mode_cb.toggled.connect(self._on_auto_mode_changed)

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
    # Range selection (Dango-style fullscreen drag)
    # ------------------------------------------------------------------
    def _on_select_range(self):
        selector = ScreenRangeSelector(self)
        selector.region_selected.connect(self._set_region)
        selector.show()

    def _set_region(self, rect: Tuple[int, int, int, int]):
        self._region = rect
        self.region_label.setText(self.tr("Region: x={} y={} w={} h={}").format(*rect))
        self.output.appendPlainText(self.tr("Region set: {}").format(rect))
        # If auto-mode is on, trigger immediately
        if self.auto_mode_cb.isChecked():
            self._on_translate_now()

    def _on_test_capture(self):
        self.output.appendPlainText("[Test Capture] starting...")
        if self._region is None:
            self.output.appendPlainText("[Test Capture] FAIL: no region selected")
            return
        img = self._capture_region()
        if img is None:
            self.output.appendPlainText("[Test Capture] FAIL: capture returned None")
            return
        if img.size == 0:
            self.output.appendPlainText("[Test Capture] FAIL: capture returned empty array")
            return
        mean_val = float(np.mean(img))
        std_val = float(np.std(img))
        self.output.appendPlainText(
            f"[Test Capture] OK: shape={img.shape}, dtype={img.dtype}, mean={mean_val:.1f}, std={std_val:.1f}"
        )
        if mean_val < 1.0 and std_val < 1.0:
            self.output.appendPlainText("[Test Capture] WARNING: image appears blank (all black)")

    # ------------------------------------------------------------------
    # QScreen capture with SSIM deduplication
    # ------------------------------------------------------------------
    def _capture_region(self) -> Optional[np.ndarray]:
        if self._region is None:
            return None
        x, y, w, h = self._region
        screen = QApplication.primaryScreen()
        if screen is None:
            self.output.appendPlainText("[Capture] ERROR: QApplication.primaryScreen() is None")
            return None
        try:
            pixmap = screen.grabWindow(0, x, y, w, h)
        except Exception as e:
            self.output.appendPlainText(f"[Capture] ERROR: grabWindow failed: {e}")
            return None
        if pixmap.isNull():
            self.output.appendPlainText("[Capture] ERROR: pixmap.isNull()")
            return None
        img = self._pixmap_to_numpy(pixmap)
        return img

    @staticmethod
    def _pixmap_to_numpy(pixmap) -> np.ndarray:
        image = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 4)
        import numpy as np
        arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))
        return arr[:, :, :3]  # drop alpha

    def _images_similar(self, img_a: np.ndarray, img_b: np.ndarray) -> bool:
        if not _CV2_OK:
            return False
        try:
            if img_a.shape != img_b.shape:
                return False
            gray_a = cvtColor(img_a, COLOR_BGR2GRAY)
            gray_b = cvtColor(img_b, COLOR_BGR2GRAY)
            score = ssim(gray_a, gray_b)
            threshold = self.ssim_threshold.value() / 100.0
            return score >= threshold
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Start / pause / auto mode
    # ------------------------------------------------------------------
    def _modules_ready(self) -> bool:
        if self._module_manager is None:
            return False
        return (
            getattr(self._module_manager, "textdetector", None) is not None
            and getattr(self._module_manager, "ocr", None) is not None
            and getattr(self._module_manager, "translator", None) is not None
        )

    def _update_start_button_state(self):
        ready = self._modules_ready() and self._region is not None
        self.start_btn.setEnabled(ready)
        if not self._modules_ready():
            self.start_btn.setToolTip(self.tr("Waiting for detector, OCR, and translator modules to load..."))
        elif self._region is None:
            self.start_btn.setToolTip(self.tr("Select a screen region first."))
        else:
            self.start_btn.setToolTip(self.tr("Start live screen capture + OCR + translation"))

    def _on_start(self):
        if not self._modules_ready():
            self.output.appendPlainText(self.tr("Cannot start: detector, OCR, or translator not loaded yet."))
            return
        if self._region is None:
            self.output.appendPlainText(self.tr("Cannot start: no region selected. Click 'Select Screen Region' first."))
            return
        self._stop_sign = False
        self._timer.start()
        self.output.appendPlainText(self.tr("Auto-translate started."))

    def _on_pause(self):
        self._timer.stop()
        self._stop_sign = True
        self.output.appendPlainText(self.tr("Auto-translate paused."))

    def _on_auto_mode_changed(self, checked: bool):
        if checked and self._region is not None and self._modules_ready():
            self._on_start()

    def _on_translate_now(self):
        self._tick()

    # ------------------------------------------------------------------
    # Tick: screenshot → SSIM skip? → background OCR/translate → result
    # ------------------------------------------------------------------
    def _tick(self):
        self.output.appendPlainText(f"[_tick] start: region={self._region}, pending_running={self._pending_worker is not None and self._pending_worker.isRunning()}")
        if self._pending_worker is not None and self._pending_worker.isRunning():
            self.output.appendPlainText("[_tick] SKIP: previous worker still running")
            return
        if self._region is None:
            self.output.appendPlainText("[_tick] SKIP: no region selected")
            return

        self.output.appendPlainText("[_tick] capturing region...")
        img = self._capture_region()
        if img is None:
            self.output.appendPlainText("[_tick] FAIL: capture returned None")
            return
        if img.size == 0:
            self.output.appendPlainText("[_tick] FAIL: capture returned empty array")
            return
        self.output.appendPlainText(f"[_tick] captured shape={img.shape} dtype={img.dtype}")

        # SSIM deduplication: skip if screen hasn't changed
        if self._last_image is not None and self._images_similar(self._last_image, img):
            self.output.appendPlainText("[_tick] SKIP: screen unchanged (SSIM)")
            return
        self._last_image = img.copy()

        self.output.appendPlainText("[_tick] starting OCR+translation worker...")
        # Run OCR+translation in background thread (Dango-style QThread subclass)
        worker = _OcrTrWorker(img, self._ocr_image, self._translate_text)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(worker.deleteLater)
        self._pending_worker = worker
        worker.start()

    def _on_worker_finished(self, state: RealtimeState):
        self._pending_worker = None
        self.output.appendPlainText(
            f"[_on_worker_finished] status={state.status}, ocr_len={len(state.last_ocr_text or '')}, tr_len={len(state.last_translation or '')}"
        )
        for w in state.warnings:
            if str(w).strip():
                self.output.appendPlainText(self.tr("Warning: {}").format(str(w).strip()))
        line = f"status={state.status} | src={state.last_ocr_text} | tr={state.last_translation}"
        self.output.appendPlainText(line)

        # Show result at first detected block position (or center of region if no blocks)
        if state.last_translation and state.status == "translated":
            self._result_window.set_text(state.last_translation)
            # Position at first block if available, otherwise at region center
            # Add region offset since block coordinates are relative to captured region
            if self._last_blocks and self._region:
                rx, ry, _, _ = self._region
                blk = self._last_blocks[0]
                x1, y1, x2, y2 = blk.xyxy
                abs_x, abs_y = int(rx + x1), int(ry + y1)
                self._result_window.show_at(abs_x, abs_y, int(x2-x1), int(y2-y1))
                self.output.appendPlainText(f"Result shown at absolute: ({abs_x}, {abs_y}, {int(x2-x1)}, {int(y2-y1)}) [block: {x1},{y1} region: {rx},{ry}]")
            elif self._region:
                x, y, w, h = self._region
                # Show at center of region, smaller size
                cx, cy = x + w//4, y + h//4
                self._result_window.show_at(cx, cy, w//2, 100)
                self.output.appendPlainText(f"Result shown at region center: ({cx}, {cy})")
        else:
            self._result_window.hide()

    # ------------------------------------------------------------------
    # OCR & Translation — wired to ModuleManager pipeline
    # ------------------------------------------------------------------
    def _ocr_image(self, img: np.ndarray) -> str:
        """Run text detection + OCR on a captured image. Returns joined text."""
        if self._module_manager is None:
            return ""
        from utils.textblock import TextBlock
        try:
            detector = getattr(self._module_manager, "textdetector", None)
            if detector is None:
                self.output.appendPlainText(self.tr("Realtime: text detector not loaded"))
                return ""
            blk_list = detector.detect(img)
            if not blk_list:
                self.output.appendPlainText(self.tr("Realtime: no text blocks detected"))
                return ""
            # Convert polygon lists/arrays to TextBlock objects if needed
            if blk_list and not isinstance(blk_list[0], TextBlock):
                def _poly_to_textblock(poly):
                    # Convert various polygon formats to xyxy bounding box
                    arr = np.array(poly)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        # Nx2 polygon -> bounding box
                        x1, y1 = arr[:, 0].min(), arr[:, 1].min()
                        x2, y2 = arr[:, 0].max(), arr[:, 1].max()
                    elif arr.ndim == 1 and arr.size == 4:
                        # Flat [x1, y1, x2, y2]
                        x1, y1, x2, y2 = arr.tolist()
                    elif arr.ndim >= 1:
                        # Fallback: flatten and take first 4
                        flat = arr.flatten()
                        x1, y1, x2, y2 = float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
                    else:
                        raise ValueError(f"Unexpected polygon format: {poly}")
                    return TextBlock([int(x1), int(y1), int(x2), int(y2)])
                blk_list = [_poly_to_textblock(blk) for blk in blk_list]
                self.output.appendPlainText(f"Realtime: wrapped {len(blk_list)} polygons into TextBlocks")
            ocr = getattr(self._module_manager, "ocr", None)
            if ocr is None:
                self.output.appendPlainText(self.tr("Realtime: OCR not loaded"))
                return ""
            ocr.run_ocr(img, blk_list)
            parts = []
            for blk in blk_list:
                txt = blk.get_text()
                if txt and str(txt).strip():
                    parts.append(str(txt).strip())
            result = "\n".join(parts)
            self.output.appendPlainText(f"Realtime: OCR extracted {len(parts)} text parts")
            # Store blocks for positioning result window
            self._last_blocks = blk_list
            return result
        except Exception as e:
            import traceback
            self.output.appendPlainText(self.tr("Realtime OCR error: {}").format(e))
            self.output.appendPlainText(traceback.format_exc())
            return ""

    def _translate_text(self, text: str) -> str:
        """Translate a plain text string using the current translator."""
        if self._module_manager is None or not text:
            return ""
        try:
            translator = getattr(self._module_manager, "translator", None)
            if translator is None:
                self.output.appendPlainText(self.tr("Realtime: translator not loaded"))
                return ""
            blk = TextBlock()
            blk.xyxy = [0, 0, 1, 1]
            blk.angle = 0
            blk.text = [text]
            blk.translation = ""
            translator.translate_textblk_lst([blk])
            return str(getattr(blk, "translation", "") or "")
        except Exception as e:
            self.output.appendPlainText(self.tr("Realtime translation error: {}").format(e))
            return ""

    def _update_min_ocr_interval(self):
        pass  # not used in this design

    def _append_backend_info(self):
        self.output.appendPlainText("capture_backend=qscreen_grabWindow")

    def closeEvent(self, event):
        self._timer.stop()
        self._ready_timer.stop()
        self._overlay.hide()
        self._result_window.hide()
        super().closeEvent(event)


class _TranslationResultWindow(QWidget):
    """Frameless always-on-top window showing translated text with Dango-style outlined QTextBrowser.
    Non-blocking: doesn't steal focus and allows mouse clicks to pass through to underlying windows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Allow mouse events to pass through to underlying windows
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self._browser = QTextBrowser(self)
        self._browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._browser.setStyleSheet(
            "border-width: 0; border-style: outset; color: white; font-weight: bold; background-color: rgba(62,62,62,180);"
        )
        layout.addWidget(self._browser)

        self._format = QTextCharFormat()
        self._format.setForeground(QColor("#FFFFFF"))
        self._format.setTextOutline(QPen(QColor(0, 120, 255), 0.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

        self._text = ""

    def set_text(self, text: str):
        self._text = str(text or "")
        self._browser.clear()
        cursor = self._browser.textCursor()
        cursor.mergeCharFormat(self._format)
        for line in self._text.split("\n"):
            cursor.insertText(line)
            cursor.insertBlock()
        self._browser.setTextCursor(cursor)
        self._resize_to_content()

    def show_at(self, x: int, y: int, w: int, h: int):
        self.setGeometry(x, y, w, h)
        self.show()
        self.raise_()
        self.activateWindow()

    def _resize_to_content(self):
        doc = self._browser.document()
        new_h = int(doc.size().height()) + 30
        self.resize(self.width(), new_h)
        self._browser.setGeometry(0, 0, self.width(), new_h)

