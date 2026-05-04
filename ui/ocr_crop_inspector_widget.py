from qtpy.QtCore import Qt, QPropertyAnimation, QEasingCurve
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QHBoxLayout, QGraphicsOpacityEffect
import numpy as np

from .misc import ndarray2pixmap


class OcrCropInspectorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(self.tr("OCR Crop Inspector")))
        row = QHBoxLayout()
        self.listw = QListWidget(self)
        self.preview = QLabel(self)
        self.preview.setMinimumWidth(280)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setText(self.tr("Select a block"))
        row.addWidget(self.listw, 1)
        row.addWidget(self.preview, 1)
        lay.addLayout(row)
        self.text_lbl = QLabel(self)
        self.text_lbl.setWordWrap(True)
        lay.addWidget(self.text_lbl)
        self._img = None
        self._blks = []
        self.listw.currentRowChanged.connect(self._on_row)

        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._anim = QPropertyAnimation(self._opacity, b"opacity", self)
        self._anim.setDuration(220)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def set_page(self, img: np.ndarray, blocks: list):
        self._img = img
        self._blks = blocks or []
        self.listw.clear()
        for i, b in enumerate(self._blks):
            conf = getattr(b, 'confidence', None)
            txt = getattr(b, 'text', '') or ''
            item = QListWidgetItem(f'#{i+1} conf={conf if conf is not None else "-"} {txt[:36]}')
            self.listw.addItem(item)
        if self._blks:
            self.listw.setCurrentRow(0)
        self._anim.start()

    def _on_row(self, row: int):
        if row < 0 or row >= len(self._blks) or self._img is None:
            return
        b = self._blks[row]
        x1,y1,x2,y2 = [int(v) for v in getattr(b, 'xyxy', [0,0,0,0])]
        h,w = self._img.shape[:2]
        x1,y1 = max(0,x1),max(0,y1)
        x2,y2 = min(w,max(x1+1,x2)),min(h,max(y1+1,y2))
        crop = self._img[y1:y2, x1:x2]
        self.preview.setPixmap(ndarray2pixmap(crop).scaledToWidth(320, Qt.TransformationMode.SmoothTransformation))
        self.text_lbl.setText(f"OCR: {getattr(b, 'text', '')}\nTranslation: {getattr(b, 'translation', '')}")
