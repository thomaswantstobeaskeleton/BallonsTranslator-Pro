from qtpy.QtCore import Signal
from qtpy.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout


class OcrTriageDialog(QDialog):
    open_block_requested = Signal(int)
    mark_open_requested = Signal(list)
    mark_reviewed_requested = Signal(list)

    def __init__(self, rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("OCR triage worklist"))
        self.resize(880, 480)
        self._rows = rows or []
        lay = QVBoxLayout(self)
        self.table = QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([self.tr("Block"), self.tr("Source"), self.tr("Translation"), self.tr("Issue")])
        self.table.setRowCount(len(self._rows))
        for r, row in enumerate(self._rows):
            self.table.setItem(r, 0, QTableWidgetItem(str(row.get("block", ""))))
            self.table.setItem(r, 1, QTableWidgetItem(str(row.get("source", ""))))
            self.table.setItem(r, 2, QTableWidgetItem(str(row.get("translation", ""))))
            self.table.setItem(r, 3, QTableWidgetItem(str(row.get("issue", ""))))
        self.table.resizeColumnsToContents()
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.doubleClicked.connect(self._on_open_selected_block)
        lay.addWidget(self.table)

        btns = QHBoxLayout()
        open_btn = QPushButton(self.tr("Open selected block"), self)
        open_btn.clicked.connect(self._on_open_selected_block)
        mark_open_btn = QPushButton(self.tr("Mark selected open"), self)
        mark_open_btn.clicked.connect(self._on_mark_selected_open)
        mark_reviewed_btn = QPushButton(self.tr("Mark selected reviewed"), self)
        mark_reviewed_btn.clicked.connect(self._on_mark_selected_reviewed)
        close_btn = QPushButton(self.tr("Close"), self)
        close_btn.clicked.connect(self.accept)
        btns.addWidget(open_btn)
        btns.addWidget(mark_open_btn)
        btns.addWidget(mark_reviewed_btn)
        btns.addStretch()
        btns.addWidget(close_btn)
        lay.addLayout(btns)

    def _selected_block_indices(self) -> list:
        out = []
        for ridx in sorted({i.row() for i in self.table.selectedIndexes()}):
            item = self.table.item(ridx, 0)
            if item is None:
                continue
            try:
                out.append(max(0, int(item.text()) - 1))
            except Exception:
                continue
        return sorted(set(out))

    def _on_open_selected_block(self):
        idxs = self._selected_block_indices()
        if idxs:
            self.open_block_requested.emit(idxs[0])

    def _on_mark_selected_open(self):
        idxs = self._selected_block_indices()
        if idxs:
            self.mark_open_requested.emit(idxs)

    def _on_mark_selected_reviewed(self):
        idxs = self._selected_block_indices()
        if idxs:
            self.mark_reviewed_requested.emit(idxs)
