from qtpy.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout


class OcrTriageDialog(QDialog):
    def __init__(self, rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("OCR triage worklist"))
        self.resize(760, 420)
        lay = QVBoxLayout(self)
        self.table = QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([self.tr("Block"), self.tr("Source"), self.tr("Translation"), self.tr("Issue")])
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(str(row.get("block", ""))))
            self.table.setItem(r, 1, QTableWidgetItem(str(row.get("source", ""))))
            self.table.setItem(r, 2, QTableWidgetItem(str(row.get("translation", ""))))
            self.table.setItem(r, 3, QTableWidgetItem(str(row.get("issue", ""))))
        self.table.resizeColumnsToContents()
        lay.addWidget(self.table)
        btns = QHBoxLayout()
        btns.addStretch()
        close_btn = QPushButton(self.tr("Close"), self)
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        lay.addLayout(btns)
