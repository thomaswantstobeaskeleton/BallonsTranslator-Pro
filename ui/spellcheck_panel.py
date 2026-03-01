"""
Spell check panel for OCR/translation text (PR #974).
Shows misspelled words and suggestions; replace on click.
"""

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QMessageBox,
)
from qtpy.QtCore import Signal, Qt


class SpellCheckPanel(QWidget):
    """Panel that lists misspellings and allows applying suggestions."""

    replace_requested = Signal(int, int, str, bool)  # block_idx, line_idx, new_line, is_translation

    def __init__(self, parent=None):
        super().__init__(parent)
        self._get_blocks = None
        self._apply_replacement = None
        self._issues = []  # list of (block_idx, line_idx, word, start, end, suggestions, text, use_translation)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr('Target:')))
        self.target_combo = QComboBox()
        self.target_combo.addItems([self.tr('Source text'), self.tr('Translation')])
        row.addWidget(self.target_combo)
        self.check_btn = QPushButton(self.tr('Check current page'))
        self.check_btn.clicked.connect(self._on_check)
        row.addWidget(self.check_btn)
        layout.addLayout(row)
        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(120)
        layout.addWidget(self.list_widget)
        self.replace_btn = QPushButton(self.tr('Replace with suggestion'))
        self.replace_btn.clicked.connect(self._on_replace)
        self.replace_btn.setEnabled(False)
        layout.addWidget(self.replace_btn)
        self.suggestion_combo = QComboBox()
        self.suggestion_combo.setMinimumWidth(180)
        layout.addWidget(self.suggestion_combo)
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)

    def set_get_blocks(self, func):
        """Set callback that returns [(block_idx, text_lines, trans_lines), ...] for current page."""
        self._get_blocks = func

    def set_apply_replacement(self, func):
        """Set callback apply_replacement(block_idx, line_idx, new_line, is_translation)."""
        self._apply_replacement = func

    def _on_check(self):
        from utils.ocr_spellcheck import get_spell_issues, _init_enchant
        if not _init_enchant():
            QMessageBox.information(
                self,
                self.tr('Spell check'),
                self.tr('Spell check requires pyenchant and a dictionary (e.g. en_US). Install: pip install pyenchant'),
            )
            return
        if not self._get_blocks:
            return
        blocks = self._get_blocks()
        use_translation = self.target_combo.currentIndex() == 1
        self._issues = []
        for block_idx, text_lines, trans_lines in blocks:
            lines = trans_lines if use_translation else text_lines
            for line_idx, line in enumerate(lines):
                text = line if isinstance(line, str) else str(line)
                for word, start, end, suggs in get_spell_issues(text):
                    self._issues.append((block_idx, line_idx, word, start, end, suggs, text, use_translation))
        self._refresh_list()

    def _refresh_list(self):
        self.list_widget.clear()
        for tup in self._issues:
            block_idx, line_idx, word, start, end, suggs, text, _ = tup
            item = QListWidgetItem(f'Block {block_idx + 1} · "{word}" → {", ".join(suggs[:5])}')
            item.setData(Qt.ItemDataRole.UserRole, tup)
            self.list_widget.addItem(item)
        self.replace_btn.setEnabled(len(self._issues) > 0)
        self.suggestion_combo.clear()
        if self._issues:
            self.list_widget.setCurrentRow(0)
            self._on_selection_changed(0)
        else:
            self._on_selection_changed(-1)

    def _on_selection_changed(self, row):
        self.replace_btn.setEnabled(row >= 0)
        self.suggestion_combo.clear()
        if row >= 0 and row < len(self._issues):
            tup = self._issues[row]
            suggs = tup[5]
            self.suggestion_combo.addItems(suggs[:20])
            self.suggestion_combo.setCurrentIndex(0)

    def _on_replace(self):
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self._issues) or not self._apply_replacement:
            return
        tup = self._issues[row]
        block_idx, line_idx, word, start, end, suggs, text, use_translation = tup
        new_word = self.suggestion_combo.currentText() if self.suggestion_combo.count() else (suggs[0] if suggs else word)
        new_line = text[:start] + new_word + text[end:]
        self._apply_replacement(block_idx, line_idx, new_line, use_translation)
        del self._issues[row]
        self._refresh_list()
