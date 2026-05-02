from qtpy.QtCore import Qt, QPropertyAnimation, QEasingCurve
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QSlider, QSpinBox, QGraphicsOpacityEffect

from .misc import ndarray2pixmap
from modules.mask_diagnostics import build_mask_diagnostics


class MaskDiagnosticsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(self.tr("Mask Diagnostics")))
        ctl = QHBoxLayout()
        self.threshold = QSlider(Qt.Orientation.Horizontal, self)
        self.threshold.setRange(0, 255)
        self.threshold.setValue(127)
        self.dilate = QSpinBox(self)
        self.dilate.setRange(0, 8)
        self.dilate.setValue(1)
        ctl.addWidget(QLabel(self.tr("Threshold")))
        ctl.addWidget(self.threshold, 1)
        ctl.addWidget(QLabel(self.tr("Dilate")))
        ctl.addWidget(self.dilate)
        lay.addLayout(ctl)
        self.raw_label = QLabel(self)
        self.th_label = QLabel(self)
        self.di_label = QLabel(self)
        for w in (self.raw_label, self.th_label, self.di_label):
            w.setMinimumHeight(120)
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(w)
        self._mask = None
        self.threshold.valueChanged.connect(self.refresh)
        self.dilate.valueChanged.connect(self.refresh)
        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._anim = QPropertyAnimation(self._opacity, b"opacity", self)
        self._anim.setDuration(260)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.start()

    def set_mask(self, mask):
        self._mask = mask
        self.refresh()

    def refresh(self):
        diag = build_mask_diagnostics(self._mask, self.threshold.value(), self.dilate.value())
        if not diag:
            for w, txt in ((self.raw_label, "Raw"), (self.th_label, "Thresholded"), (self.di_label, "Dilated")):
                w.setText(self.tr(txt))
            return
        self.raw_label.setPixmap(ndarray2pixmap(diag["raw"]).scaledToWidth(320, Qt.TransformationMode.SmoothTransformation))
        self.th_label.setPixmap(ndarray2pixmap(diag["thresholded"]).scaledToWidth(320, Qt.TransformationMode.SmoothTransformation))
        self.di_label.setPixmap(ndarray2pixmap(diag["dilated"]).scaledToWidth(320, Qt.TransformationMode.SmoothTransformation))
