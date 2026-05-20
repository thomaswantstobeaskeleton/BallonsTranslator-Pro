from typing import List, Dict

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QComboBox, QPlainTextEdit
)


class CompareResultsDialog(QDialog):
    def __init__(self, scope: str, candidates: List[Dict], parent=None):
        super().__init__(parent)
        self._scope = str(scope or "translator")
        self._candidates = list(candidates or [])
        self._visible_rows = list(range(len(self._candidates)))

        self.setWindowTitle(self.tr("Compare results"))
        self.resize(1040, 560)

        lay = QVBoxLayout(self)
        hint = QLabel(self.tr("Select one candidate row and click Apply. Double-click also applies."), self)
        hint.setWordWrap(True)
        lay.addWidget(hint)

        tool_row = QHBoxLayout()
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText(self.tr("Filter by provider or candidate text..."))
        self.search_edit.textChanged.connect(self._rebuild_table)
        self.provider_filter = QComboBox(self)
        self.provider_filter.addItem(self.tr("All providers"), "")
        for name in sorted({str((r or {}).get("provider", "") or "") for r in self._candidates if str((r or {}).get("provider", "") or "").strip()}):
            self.provider_filter.addItem(name, name)
        self.provider_filter.currentIndexChanged.connect(self._rebuild_table)
        tool_row.addWidget(QLabel(self.tr("Filter"), self))
        tool_row.addWidget(self.search_edit, 1)
        tool_row.addWidget(self.provider_filter)
        lay.addLayout(tool_row)

        self.table = QTableWidget(self)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            self.tr("Pick"), self.tr("Provider"), self.tr("Candidate"), self.tr("Latency (ms)"), self.tr("Source"), self.tr("Current")
        ])
        self.table.setRowCount(len(self._candidates))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.doubleClicked.connect(self.accept)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self._rebuild_table()

        lay.addWidget(self.table)

        self.preview = QPlainTextEdit(self)
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText(self.tr("Candidate preview"))
        self.preview.setMinimumHeight(120)
        lay.addWidget(self.preview)

        btns = QHBoxLayout()
        self.copy_btn = QPushButton(self.tr("Copy selected text"), self)
        self.copy_btn.clicked.connect(self._copy_selected_text)
        self.apply_btn = QPushButton(self.tr("Apply selected"), self)
        self.apply_btn.clicked.connect(self.accept)
        close_btn = QPushButton(self.tr("Cancel"), self)
        close_btn.clicked.connect(self.reject)
        btns.addWidget(self.copy_btn)
        btns.addWidget(self.apply_btn)
        btns.addStretch(1)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

    def _rebuild_table(self):
        needle = str(self.search_edit.text() or "").strip().lower() if hasattr(self, "search_edit") else ""
        provider_sel = str(self.provider_filter.currentData() or "").strip().lower() if hasattr(self, "provider_filter") else ""
        self._visible_rows = []
        for src_idx, row in enumerate(self._candidates):
            tele = dict(row.get("telemetry", {}) or {})
            provider = str(row.get("provider", "") or "")
            cand = str(row.get("text", "") or "")
            if needle and needle not in provider.lower() and needle not in cand.lower():
                continue
            if provider_sel and str(row.get("provider", "") or "").strip().lower() != provider_sel:
                continue
            self._visible_rows.append(src_idx)
        self.table.setRowCount(len(self._visible_rows))
        for r, src_idx in enumerate(self._visible_rows):
            row = self._candidates[src_idx]
            tele = dict(row.get("telemetry", {}) or {})
            provider = str(row.get("provider", "") or "")
            cand = str(row.get("text", "") or "")
            lat = str(int(tele.get("latency_ms", 0) or 0))
            src = str(tele.get("source", "") or "")
            current = self.tr("Yes") if bool(tele.get("is_current", False)) else ""

            self.table.setItem(r, 0, QTableWidgetItem(str(src_idx + 1)))
            self.table.setItem(r, 1, QTableWidgetItem(provider))
            self.table.setItem(r, 2, QTableWidgetItem(cand))
            self.table.setItem(r, 3, QTableWidgetItem(lat))
            self.table.setItem(r, 4, QTableWidgetItem(src))
            self.table.setItem(r, 5, QTableWidgetItem(current))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        if self.table.rowCount() > 0:
            self.table.selectRow(0)
        self._on_selection_changed()

    def _on_selection_changed(self):
        idx = self.selected_index()
        if idx < 0 or idx >= len(self._candidates):
            self.preview.setPlainText("")
            return
        row = self._candidates[idx]
        self.preview.setPlainText(str(row.get("text", "") or ""))

    def _copy_selected_text(self):
        idx = self.selected_index()
        if idx < 0 or idx >= len(self._candidates):
            return
        txt = str(self._candidates[idx].get("text", "") or "")
        try:
            from qtpy.QtWidgets import QApplication
            QApplication.clipboard().setText(txt)
        except Exception:
            pass

    def selected_index(self) -> int:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            return -1
        view_row = rows[0]
        if view_row < 0 or view_row >= len(self._visible_rows):
            return -1
        return self._visible_rows[view_row]
