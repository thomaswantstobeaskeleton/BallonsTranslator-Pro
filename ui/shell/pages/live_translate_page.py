"""Live Translate page — launcher for the existing realtime translator."""

from __future__ import annotations
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QPlainTextEdit, QListWidget, QListWidgetItem, QSplitter, QMessageBox,
)
from qtpy.QtCore import Qt, Signal

from utils.config import pcfg
from ..theme import COLORS, SPACING
from .components import ShellCard, PageHeader, AccentButton, StatusPill


class LiveTranslatePage(QWidget):
    """Live translate launcher page. When dev_mode is enabled, this can
    open the existing RealtimeTranslatorDialog for screen-region translation."""

    open_translator_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        root.setSpacing(SPACING.xl)
        root.addWidget(PageHeader(
            "Live Translate",
            "Screen-region OCR and translation. Requires Developer Mode."
        ))

        dev_card = ShellCard("Developer Mode Status")
        dev_row = QHBoxLayout()
        is_dev = getattr(pcfg, 'dev_mode', False)
        dev_row.addWidget(QLabel("Developer Mode"))
        dev_row.addStretch()
        dev_row.addWidget(StatusPill(
            "Enabled" if is_dev else "Disabled",
            COLORS.success if is_dev else COLORS.warning
        ))
        dev_card.layout.addLayout(dev_row)
        if not is_dev:
            info = QLabel(
                "Enable Developer Mode in Settings > General to use Live Translate."
            )
            info.setStyleSheet(f"color: {COLORS.text_secondary}; background: transparent;")
            dev_card.layout.addWidget(info)
        root.addWidget(dev_card)

        top = QHBoxLayout()
        for label, values in [
            ("Source", ["Japanese", "Chinese", "Korean", "Auto Detect"]),
            ("Target", ["English", "French", "German", "Spanish"]),
            ("Provider", ["DeepL", "Google", "OpenAI", "Local"]),
        ]:
            card = ShellCard(label)
            cb = QComboBox()
            cb.addItems(values)
            card.layout.addWidget(cb)
            top.addWidget(card)
        root.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        input_card = ShellCard("Input")
        self._input_box = QPlainTextEdit()
        self._input_box.setPlaceholderText("Paste text here for quick translation...")
        input_card.layout.addWidget(self._input_box)
        translate_btn = AccentButton("Translate")
        translate_btn.clicked.connect(self._quick_translate)
        input_card.layout.addWidget(translate_btn)
        splitter.addWidget(input_card)

        history_card = ShellCard("History")
        self._history = QListWidget()
        history_card.layout.addWidget(self._history)
        splitter.addWidget(history_card)
        splitter.setSizes([520, 420])
        root.addWidget(splitter, 1)

        start_btn = AccentButton("Open Realtime Translator")
        start_btn.clicked.connect(self._open_translator)
        root.addWidget(start_btn)

    def _quick_translate(self):
        text = self._input_box.toPlainText().strip()
        if text:
            self._history.addItem(QListWidgetItem(f"[Input] {text[:60]}..."))
            # TODO: wire to actual translation module

    def _open_translator(self):
        if not getattr(pcfg, 'dev_mode', False):
            QMessageBox.information(
                self, "Developer Mode Required",
                "Enable Developer Mode in Settings > General to use the Realtime Translator."
            )
            return
        self.open_translator_requested.emit()
