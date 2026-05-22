from __future__ import annotations
from qtpy.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QPlainTextEdit, QCheckBox, QSpinBox, QFormLayout
from qtpy.QtCore import QTimer
import numpy as np

from utils.realtime_mode import RealtimeRegion, RealtimeWatcher, NumpyFrameBackend, create_screenshot_backend, OverlayExclusionBackend
from utils.config import pcfg, save_config


class RealtimeTranslatorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Realtime Screen Translator"))
        self.resize(520, 420)
        root = QVBoxLayout(self)
        row = QHBoxLayout()
        self.profile_combo = QComboBox(self)
        self.profile_combo.addItem(self.tr("Chrome Manhua Reader"), "chrome_manhua_reader")
        self.profile_combo.addItem(self.tr("Manga Reader"), "manga_reader")
        self.profile_combo.addItem(self.tr("Generic Screen OCR"), "generic_screen_ocr")
        _profile = str(getattr(pcfg, "realtime_profile_id", "generic_screen_ocr") or "generic_screen_ocr")
        _pidx = self.profile_combo.findData(_profile)
        self.profile_combo.setCurrentIndex(_pidx if _pidx >= 0 else 2)
        self.ocr_combo = QComboBox(self)
        self.ocr_combo.addItems(["auto", "manga_ocr", "paddle", "llm_ocr"])
        self.tr_combo = QComboBox(self)
        self.tr_combo.addItems(["google", "deepl", "openai", "ollama"])
        row.addWidget(QLabel(self.tr("Profile"), self))
        row.addWidget(self.profile_combo)
        row.addWidget(QLabel(self.tr("OCR"), self))
        row.addWidget(self.ocr_combo)
        row.addWidget(QLabel(self.tr("Translator"), self))
        row.addWidget(self.tr_combo)
        root.addLayout(row)

        self.privacy_hint = QLabel(self.tr("Privacy defaults: screenshots/OCR/translation are not persisted or logged."), self)
        self.privacy_hint.setWordWrap(True)
        root.addWidget(self.privacy_hint)

        self.follow_window = QCheckBox(self.tr("Follow selected window (when available)"), self)
        self.follow_window.setChecked(bool(getattr(pcfg, "realtime_follow_window", False)))
        self.follow_window.toggled.connect(lambda *_: self._persist_settings())
        root.addWidget(self.follow_window)
        self.status_hint = QLabel(self.tr("Status: idle"), self)
        self.status_hint.setStyleSheet("color: gray;")
        root.addWidget(self.status_hint)

        form = QFormLayout()
        self.x_spin = QSpinBox(self); self.x_spin.setRange(0, 10000)
        self.y_spin = QSpinBox(self); self.y_spin.setRange(0, 10000)
        self.w_spin = QSpinBox(self); self.w_spin.setRange(1, 10000); self.w_spin.setValue(100)
        self.h_spin = QSpinBox(self); self.h_spin.setRange(1, 10000); self.h_spin.setValue(100)
        _rect = list(getattr(pcfg, "realtime_region_rect", [0, 0, 100, 100]) or [0, 0, 100, 100])
        while len(_rect) < 4:
            _rect.append(0)
        self.x_spin.setValue(int(_rect[0]))
        self.y_spin.setValue(int(_rect[1]))
        self.w_spin.setValue(max(1, int(_rect[2] or 100)))
        self.h_spin.setValue(max(1, int(_rect[3] or 100)))
        form.addRow(self.tr("Region X"), self.x_spin)
        form.addRow(self.tr("Region Y"), self.y_spin)
        form.addRow(self.tr("Region Width"), self.w_spin)
        form.addRow(self.tr("Region Height"), self.h_spin)

        self.capture_interval_ms = QSpinBox(self); self.capture_interval_ms.setRange(100, 5000); self.capture_interval_ms.setValue(int(getattr(pcfg, "realtime_capture_interval_ms", 700) or 700))
        self.min_ocr_interval_ms = QSpinBox(self); self.min_ocr_interval_ms.setRange(0, 5000); self.min_ocr_interval_ms.setValue(int(getattr(pcfg, "realtime_min_ocr_interval_ms", 0) or 0))
        form.addRow(self.tr("Capture interval (ms)"), self.capture_interval_ms)
        form.addRow(self.tr("Min OCR interval (ms)"), self.min_ocr_interval_ms)
        root.addLayout(form)

        btns = QHBoxLayout()
        self.start_btn = QPushButton(self.tr("Start"), self)
        self.pause_btn = QPushButton(self.tr("Pause"), self)
        self.now_btn = QPushButton(self.tr("Translate now"), self)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.pause_btn)
        btns.addWidget(self.now_btn)
        root.addLayout(btns)

        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        root.addWidget(self.output)

        self._overlay = QLabel("", None)
        self._overlay.setWindowTitle("Realtime Overlay")
        self._overlay.setStyleSheet("background: rgba(0,0,0,180); color: white; padding: 8px;")
        self._overlay.hide()

        self._backend = create_screenshot_backend("auto")
        self._fallback_backend = NumpyFrameBackend([np.zeros((20, 40, 3), dtype=np.uint8)])
        self._wrapped_backend = OverlayExclusionBackend(self._backend, self._hide_overlay_for_capture, self._show_overlay_after_capture)
        self._watcher = RealtimeWatcher(self._wrapped_backend, lambda img: "示例文本" if img.mean() > 1 else "", lambda t: "sample translation")
        self._region = RealtimeRegion("default", (0, 0, 100, 100))
        self._timer = QTimer(self)
        self._timer.setInterval(self.capture_interval_ms.value())
        self._timer.timeout.connect(self._tick)
        self.capture_interval_ms.valueChanged.connect(self._timer.setInterval)
        self.profile_combo.currentIndexChanged.connect(lambda *_: self._persist_settings())
        self.min_ocr_interval_ms.valueChanged.connect(self._update_min_ocr_interval)
        self.x_spin.valueChanged.connect(lambda *_: self._persist_settings())
        self.y_spin.valueChanged.connect(lambda *_: self._persist_settings())
        self.w_spin.valueChanged.connect(lambda *_: self._persist_settings())
        self.h_spin.valueChanged.connect(lambda *_: self._persist_settings())

        self.start_btn.clicked.connect(self._start_live)
        self.pause_btn.clicked.connect(self._pause_live)
        self.now_btn.clicked.connect(self._tick)
        self._append_backend_info()

    def _start_live(self):
        self._persist_settings()
        self._timer.start()
        self.status_hint.setText(self.tr("Status: running"))

    def _pause_live(self):
        self._timer.stop()
        self.status_hint.setText(self.tr("Status: paused"))

    def _tick(self):
        self._region.profile = str(self.profile_combo.currentData() or "generic_screen_ocr")
        self._region.follow_window = bool(self.follow_window.isChecked())
        self._region.rect = (self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value())
        st = self._watcher.tick(self._region)
        line = f"status={st.status} | src={st.last_ocr_text} | tr={st.last_translation}"
        self.output.appendPlainText(line)
        self.status_hint.setText(self.tr("Status: {0}").format(st.status))
        if st.last_translation:
            self._overlay.setText(st.last_translation)
            self._overlay.adjustSize()
            self._overlay.show()

    def _update_min_ocr_interval(self):
        self._watcher.min_ocr_interval_sec = float(self.min_ocr_interval_ms.value()) / 1000.0
        self._persist_settings()

    def _append_backend_info(self):
        primary = getattr(self._wrapped_backend, "backend_name", "unknown")
        fallback = getattr(self._fallback_backend, "backend_name", "unknown")
        self.output.appendPlainText(f"capture_backend={primary} | fallback={fallback}")
        if not bool(getattr(self._wrapped_backend, "supports_window_exclusion", False)):
            self.output.appendPlainText("overlay_exclusion=fallback_hide_show")

    def _hide_overlay_for_capture(self):
        if self._overlay.isVisible():
            self._overlay.hide()

    def _show_overlay_after_capture(self):
        if self._timer.isActive() and bool(getattr(self._watcher, "state_by_region", None)):
            self._overlay.show()

    def _persist_settings(self):
        pcfg.realtime_profile_id = str(self.profile_combo.currentData() or "generic_screen_ocr")
        pcfg.realtime_capture_interval_ms = int(self.capture_interval_ms.value())
        pcfg.realtime_min_ocr_interval_ms = int(self.min_ocr_interval_ms.value())
        pcfg.realtime_follow_window = bool(self.follow_window.isChecked())
        pcfg.realtime_region_rect = [int(self.x_spin.value()), int(self.y_spin.value()), int(self.w_spin.value()), int(self.h_spin.value())]
        save_config()
