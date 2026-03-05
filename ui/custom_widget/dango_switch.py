# Dango-style animated pill switch (inspired by PantsuDango/Dango-Translator ui/switch.py).
# Rounded pill background, circular slider, gradient when ON, QTimer-based slide animation.

from qtpy.QtWidgets import QWidget
from qtpy.QtCore import Qt, Signal, QRectF, QTimer
from qtpy.QtGui import QPainter, QColor, QLinearGradient, QPainterPath


class DangoSwitch(QWidget):
    """Pill-shaped toggle switch with animated slider and gradient when ON (Dango-style)."""

    checkedChanged = Signal(bool)

    def __init__(self, parent=None, checked=False):
        super().__init__(parent)
        self._checked = bool(checked)
        self._space = 2
        self._step = 0.0
        self._start_x = 0.0
        self._end_x = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_value)

        # Colors (Dango-inspired)
        self._bg_off = QColor("#f0f0f0")
        self._bg_on_start = QColor("#83AAF9")
        self._bg_on_end = QColor("#5B8FF9")
        self._slider_off = QColor("#fefefe")
        self._slider_on = QColor("#fefefe")
        self._text_off = QColor(143, 143, 143)
        self._text_on = QColor(255, 255, 255)

        self.setMinimumSize(52, 26)
        self.setMaximumHeight(32)
        self._recompute_end_x()

    def _recompute_end_x(self):
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        self._step = w / 50.0
        if self._checked:
            self._end_x = float(w - h)
            if self._start_x > self._end_x:
                self._start_x = self._end_x
        else:
            self._end_x = 0.0
            if self._start_x < self._end_x:
                self._start_x = self._end_x

    def _update_value(self):
        if self._checked:
            if self._start_x < self._end_x:
                self._start_x = min(self._start_x + self._step, self._end_x)
            else:
                self._start_x = self._end_x
                self._timer.stop()
        else:
            if self._start_x > self._end_x:
                self._start_x = max(self._start_x - self._step, self._end_x)
            else:
                self._start_x = self._end_x
                self._timer.stop()
        self.update()

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        if self._checked == bool(checked):
            return
        self._checked = bool(checked)
        self._recompute_end_x()
        self._timer.start(5)
        self.checkedChanged.emit(self._checked)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._recompute_end_x()
        if not self._timer.isActive():
            self._start_x = self._end_x

    def showEvent(self, event):
        super().showEvent(event)
        if not self._timer.isActive() and self.width() > 0:
            self._recompute_end_x()
            self._start_x = self._end_x

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self._draw_bg(painter)
        self._draw_slider(painter)
        self._draw_text(painter)

    def _draw_bg(self, painter):
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        rect = self.rect()
        h = rect.height()
        r = h / 2.0
        circle_w = h
        if self._checked:
            grad = QLinearGradient(0, 0, rect.width(), rect.height())
            grad.setColorAt(0, self._bg_on_start)
            grad.setColorAt(1, self._bg_on_end)
            painter.setBrush(grad)
        else:
            painter.setBrush(self._bg_off)
        path = QPainterPath()
        path.moveTo(r, rect.top())
        path.arcTo(QRectF(rect.left(), rect.top(), circle_w, circle_w), 90, 180)
        path.lineTo(rect.width() - r, rect.height())
        path.arcTo(QRectF(rect.width() - circle_w, rect.top(), circle_w, circle_w), 270, 180)
        path.lineTo(r, rect.top())
        painter.drawPath(path)
        painter.restore()

    def _draw_slider(self, painter):
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        if self._checked:
            painter.setBrush(self._slider_on)
        else:
            painter.setBrush(self._slider_off)
        rect = self.rect()
        h = rect.height()
        sw = h - self._space * 2
        sx = self._start_x + self._space
        sy = self._space
        painter.drawEllipse(QRectF(sx, sy, sw, sw))
        painter.restore()

    def _draw_text(self, painter):
        painter.save()
        rect = self.rect()
        half = rect.width() / 2
        if self._checked:
            painter.setPen(self._text_on)
            painter.drawText(
                self._space, 0,
                int(half + self._space * 2), rect.height(),
                Qt.AlignmentFlag.AlignCenter,
                self.tr("On")
            )
        else:
            painter.setPen(self._text_off)
            painter.drawText(
                int(half - self._space * 2), 0,
                int(half - self._space), rect.height(),
                Qt.AlignmentFlag.AlignCenter,
                self.tr("Off")
            )
        painter.restore()
