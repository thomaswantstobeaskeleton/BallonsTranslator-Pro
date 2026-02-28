"""
Keyboard shortcuts dialog: list all actions, edit keybindings, reset to default, apply.
"""
from typing import Dict, List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QKeySequenceEdit,
    QLineEdit,
    QComboBox,
    QPushButton,
    QLabel,
    QHeaderView,
    QAbstractItemView,
    QMessageBox,
    QWidget,
)

from utils.shortcuts import get_shortcut_info, get_default_shortcuts
from utils.config import pcfg, save_config


class ShortcutsDialog(QDialog):
    """Dialog to view and customize keyboard shortcuts."""

    shortcuts_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Keyboard Shortcuts"))
        self.setMinimumSize(620, 480)
        self.resize(700, 520)

        self._info = get_shortcut_info()  # (id, default_key, category, description)
        self._defaults = get_default_shortcuts()
        self._current: Dict[str, str] = dict(getattr(pcfg, "shortcuts", None) or self._defaults)

        layout = QVBoxLayout(self)

        # Filter row
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel(self.tr("Filter:")))
        self.filter_edit = QLineEdit(self)
        self.filter_edit.setPlaceholderText(self.tr("Search action or category..."))
        self.filter_edit.textChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.filter_edit)
        self.category_combo = QComboBox(self)
        self.category_combo.addItem(self.tr("All categories"), None)
        categories = sorted(set(item[2] for item in self._info))
        for cat in categories:
            self.category_combo.addItem(cat, cat)
        self.category_combo.currentIndexChanged.connect(self._apply_filter)
        filter_layout.addWidget(QLabel(self.tr("Category:")))
        filter_layout.addWidget(self.category_combo)
        layout.addLayout(filter_layout)

        # Table: Category | Action | Shortcut
        self.table = QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            self.tr("Category"),
            self.tr("Action"),
            self.tr("Shortcut"),
            "",  # Reset button column
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(3, 80)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.reset_all_btn = QPushButton(self.tr("Reset all to default"), self)
        self.reset_all_btn.clicked.connect(self._reset_all)
        btn_layout.addWidget(self.reset_all_btn)
        self.apply_btn = QPushButton(self.tr("Apply"), self)
        self.apply_btn.clicked.connect(self._apply)
        self.apply_btn.setDefault(True)
        btn_layout.addWidget(self.apply_btn)
        cancel_btn = QPushButton(self.tr("Cancel"), self)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self._populate_table()
        self._apply_filter()

    def _populate_table(self):
        self.table.setRowCount(len(self._info))
        self._key_edits: List[QKeySequenceEdit] = []
        for row, (action_id, default_key, category, description) in enumerate(self._info):
            self.table.setItem(row, 0, QTableWidgetItem(category))
            self.table.setItem(row, 1, QTableWidgetItem(description))
            self.table.item(row, 0).setData(Qt.ItemDataRole.UserRole, action_id)
            self.table.item(row, 1).setData(Qt.ItemDataRole.UserRole, action_id)

            key_edit = QKeySequenceEdit(self)
            key_edit.setClearButtonEnabled(True)
            key = self._current.get(action_id, default_key)
            if key:
                key_edit.setKeySequence(QKeySequence.fromString(key))
            key_edit.setMaximumWidth(180)
            key_edit.setProperty("action_id", action_id)
            key_edit.setProperty("default_key", default_key)
            self._key_edits.append(key_edit)
            self.table.setCellWidget(row, 2, key_edit)

            reset_btn = QPushButton(self.tr("Reset"), self)
            reset_btn.setProperty("action_id", action_id)
            reset_btn.setProperty("default_key", default_key)
            reset_btn.setProperty("row", row)
            reset_btn.clicked.connect(self._reset_one)
            self.table.setCellWidget(row, 3, reset_btn)

    def _reset_one(self):
        btn = self.sender()
        if not isinstance(btn, QPushButton):
            return
        action_id = btn.property("action_id")
        default_key = btn.property("default_key")
        row = btn.property("row")
        self._current[action_id] = default_key
        key_edit = self.table.cellWidget(row, 2)
        if isinstance(key_edit, QKeySequenceEdit):
            key_edit.setKeySequence(QKeySequence.fromString(default_key) if default_key else QKeySequence())

    def _reset_all(self):
        for action_id, default_key in self._defaults.items():
            self._current[action_id] = default_key
        for row in range(self.table.rowCount()):
            key_edit = self.table.cellWidget(row, 2)
            if isinstance(key_edit, QKeySequenceEdit):
                default_key = key_edit.property("default_key")
                key_edit.setKeySequence(QKeySequence.fromString(default_key) if default_key else QKeySequence())

    def _apply_filter(self):
        filter_text = self.filter_edit.text().strip().lower()
        cat_value = self.category_combo.currentData()
        for row in range(self.table.rowCount()):
            cat_item = self.table.item(row, 0)
            desc_item = self.table.item(row, 1)
            action_id = cat_item.data(Qt.ItemDataRole.UserRole)
            cat = cat_item.text()
            desc = desc_item.text()
            show = True
            if cat_value and cat != cat_value:
                show = False
            if filter_text:
                show = show and (
                    filter_text in cat.lower()
                    or filter_text in desc.lower()
                    or filter_text in action_id.lower()
                )
            self.table.setRowHidden(row, not show)

    def _apply(self):
        # Collect current keys from key edits
        for row in range(self.table.rowCount()):
            key_edit = self.table.cellWidget(row, 2)
            if isinstance(key_edit, QKeySequenceEdit):
                action_id = key_edit.property("action_id")
                seq = key_edit.keySequence()
                self._current[action_id] = seq.toString(QKeySequence.SequenceFormat.PortableText) if not seq.isEmpty() else ""

        if not isinstance(pcfg.shortcuts, dict):
            pcfg.shortcuts = {}
        pcfg.shortcuts.clear()
        pcfg.shortcuts.update(self._current)
        save_config()
        self.shortcuts_changed.emit()
        self.accept()

    def get_current_shortcuts(self) -> Dict[str, str]:
        """Return current key map (from edits, not yet applied)."""
        out = dict(self._current)
        for row in range(self.table.rowCount()):
            key_edit = self.table.cellWidget(row, 2)
            if isinstance(key_edit, QKeySequenceEdit):
                action_id = key_edit.property("action_id")
                seq = key_edit.keySequence()
                out[action_id] = seq.toString(QKeySequence.SequenceFormat.PortableText) if not seq.isEmpty() else ""
        return out
