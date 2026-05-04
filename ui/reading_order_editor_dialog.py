from qtpy.QtCore import Qt, QPropertyAnimation, QEasingCurve
from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QHBoxLayout, QPushButton, QGraphicsOpacityEffect


class ReadingOrderEditorDialog(QDialog):
    def __init__(self, blocks: list, on_commit, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Reading Order Editor"))
        self.resize(560, 520)
        self._blocks = list(blocks or [])
        self._on_commit = on_commit

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(self.tr("Reorder OCR blocks for reading/translation sequence")))
        self.listw = QListWidget(self)
        self.listw.setSelectionMode(QListWidget.SingleSelection)
        lay.addWidget(self.listw, 1)

        btn_row = QHBoxLayout()
        self.up_btn = QPushButton(self.tr("Move Up"), self)
        self.down_btn = QPushButton(self.tr("Move Down"), self)
        self.commit_btn = QPushButton(self.tr("Apply Order"), self)
        btn_row.addWidget(self.up_btn)
        btn_row.addWidget(self.down_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.commit_btn)
        lay.addLayout(btn_row)

        self.up_btn.clicked.connect(self._move_up)
        self.down_btn.clicked.connect(self._move_down)
        self.commit_btn.clicked.connect(self._commit)

        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._anim = QPropertyAnimation(self._opacity, b"opacity", self)
        self._anim.setDuration(220)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.start()

        self._refresh()

    def _refresh(self):
        self.listw.clear()
        for i, b in enumerate(self._blocks):
            text = (getattr(b, 'text', '') or '').strip().replace('\n', ' ')
            self.listw.addItem(QListWidgetItem(f"#{i+1}  {text[:70]}"))
        if self.listw.count() > 0 and self.listw.currentRow() < 0:
            self.listw.setCurrentRow(0)

    def _move_up(self):
        r = self.listw.currentRow()
        if r <= 0:
            return
        self._blocks[r - 1], self._blocks[r] = self._blocks[r], self._blocks[r - 1]
        self._refresh()
        self.listw.setCurrentRow(r - 1)

    def _move_down(self):
        r = self.listw.currentRow()
        if r < 0 or r >= len(self._blocks) - 1:
            return
        self._blocks[r + 1], self._blocks[r] = self._blocks[r], self._blocks[r + 1]
        self._refresh()
        self.listw.setCurrentRow(r + 1)

    def _commit(self):
        self._on_commit(self._blocks)
        self.accept()
