"""Command Palette foundation (mockup section 11)."""

from __future__ import annotations
from typing import Callable, Dict, List, Optional

from qtpy.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QListWidget, QListWidgetItem
from qtpy.QtCore import Qt, Signal

from .theme import COLORS, FONTS, SPACING, RADIUS
from .nav_controller import SECTION_LABELS


class CommandPalette(QDialog):
    command_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Command Palette")
        self.setModal(False)
        self.resize(520, 420)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {COLORS.bg_elevated}; border: 1px solid {COLORS.border}; border-radius: {RADIUS.lg}px; }}
            QLineEdit {{ font-size: {FONTS.size_lg}px; padding: 10px 12px; }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(SPACING.lg, SPACING.lg, SPACING.lg, SPACING.lg)
        lay.setSpacing(SPACING.md)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search commands…")
        self.list = QListWidget()
        lay.addWidget(self.search)
        lay.addWidget(self.list, 1)
        self._commands: List[tuple[str, str]] = []
        self.search.textChanged.connect(self._filter)
        self.list.itemActivated.connect(self._activate)

    def set_commands(self, commands: List[tuple[str, str]]):
        self._commands = list(commands)
        self._filter(self.search.text())

    def _filter(self, text: str):
        q = (text or "").strip().lower()
        self.list.clear()
        for label, command_id in self._commands:
            if not q or q in label.lower():
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, command_id)
                self.list.addItem(item)

    def _activate(self, item: QListWidgetItem):
        command_id = item.data(Qt.ItemDataRole.UserRole)
        if command_id:
            self.command_selected.emit(str(command_id))
            self.hide()
