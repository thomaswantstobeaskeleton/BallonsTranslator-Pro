from __future__ import annotations
from qtpy.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QPlainTextEdit, QCheckBox
from qtpy.QtCore import QTimer
import numpy as np

from utils.realtime_mode import RealtimeRegion, RealtimeWatcher, NumpyFrameBackend


class RealtimeTranslatorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Realtime Screen Translator"))
        self.resize(520, 420)
        root = QVBoxLayout(self)
        row = QHBoxLayout()
        self.ocr_combo = QComboBox(self)
        self.ocr_combo.addItems(["auto", "manga_ocr", "paddle", "llm_ocr"])
        self.tr_combo = QComboBox(self)
        self.tr_combo.addItems(["google", "deepl", "openai", "ollama"])
        row.addWidget(QLabel(self.tr("OCR"), self))
        row.addWidget(self.ocr_combo)
        row.addWidget(QLabel(self.tr("Translator"), self))
        row.addWidget(self.tr_combo)
        root.addLayout(row)

        self.privacy_hint = QLabel(self.tr("Privacy defaults: screenshots/OCR/translation are not persisted or logged."), self)
        self.privacy_hint.setWordWrap(True)
        root.addWidget(self.privacy_hint)

        self.follow_window = QCheckBox(self.tr("Follow selected window (when available)"), self)
        root.addWidget(self.follow_window)

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

        frames = [np.zeros((20, 40, 3), dtype=np.uint8), np.ones((20, 40, 3), dtype=np.uint8) * 255]
        self._watcher = RealtimeWatcher(NumpyFrameBackend(frames), lambda img: "示例文本" if img.mean() > 1 else "", lambda t: "sample translation")
        self._region = RealtimeRegion("default", (0, 0, 100, 100))
        self._timer = QTimer(self)
        self._timer.setInterval(700)
        self._timer.timeout.connect(self._tick)

        self.start_btn.clicked.connect(lambda: self._timer.start())
        self.pause_btn.clicked.connect(lambda: self._timer.stop())
        self.now_btn.clicked.connect(self._tick)

    def _tick(self):
        st = self._watcher.tick(self._region)
        line = f"status={st.status} | src={st.last_ocr_text} | tr={st.last_translation}"
        self.output.appendPlainText(line)
        if st.last_translation:
            self._overlay.setText(st.last_translation)
            self._overlay.adjustSize()
            self._overlay.show()
