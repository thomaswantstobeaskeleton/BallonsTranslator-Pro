# Focus ring animation: draw an optional focus ring that fades in/out with focus.
# Use as a wrapper around any QWidget; when the child gets/loses focus, ring opacity animates.
# When pcfg.reduce_motion is True, duration is 0.

from qtpy.QtWidgets import QWidget, QFrame, QVBoxLayout
from qtpy.QtCore import QPropertyAnimation, Qt, Property, QEvent, QEasingCurve
from qtpy.QtGui import QPainter, QColor, QPen

try:
    from utils.config import pcfg
except Exception:
    pcfg = None


def _focus_ring_duration_ms() -> int:
    """Return 0 when reduce_motion is on, else 120."""
    if pcfg is None:
        return 120
    return 0 if getattr(pcfg, 'reduce_motion', False) else 120


class FocusRingFrame(QFrame):
    """
    Wraps a child widget and draws a focus ring (rounded rect border) that fades in
    when the child gains focus and fades out when it loses focus.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ring_opacity = 0.0
        self._anim = QPropertyAnimation(self, b"ring_opacity")
        self._anim.setDuration(120)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.setParent(self)
        self._child = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    def setChild(self, child: QWidget):
        """Set the widget to wrap and watch for focus. Call once after child is created."""
        if self._child is not None:
            self._child.removeEventFilter(self)
        self._child = child
        if self.layout() is None:
            lay = QVBoxLayout(self)
            lay.setContentsMargins(2, 2, 2, 2)
        self.layout().addWidget(child)
        child.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self._child and event.type() == QEvent.Type.FocusIn:
            self._anim.stop()
            self._anim.setDuration(_focus_ring_duration_ms())
            self._anim.setStartValue(self._ring_opacity)
            self._anim.setEndValue(1.0)
            self._anim.start()
        elif obj == self._child and event.type() == QEvent.Type.FocusOut:
            self._anim.stop()
            self._anim.setDuration(_focus_ring_duration_ms())
            self._anim.setStartValue(self._ring_opacity)
            self._anim.setEndValue(0.0)
            self._anim.start()
        return False

    def get_ring_opacity(self):
        return self._ring_opacity

    def set_ring_opacity(self, value: float):
        self._ring_opacity = max(0.0, min(1.0, value))
        self.update()

    ring_opacity = Property(float, get_ring_opacity, set_ring_opacity)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._ring_opacity <= 0.0 or self._child is None:
            return
        painter = QPainter(self)
        if not painter.begin(self):
            return
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            r = self.rect().adjusted(1, 1, -1, -1)
            pen = QPen(QColor(100, 149, 237))  # cornflower blue; could use palette
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setOpacity(self._ring_opacity)
            painter.drawRoundedRect(r, 4, 4)
        finally:
            painter.end()
