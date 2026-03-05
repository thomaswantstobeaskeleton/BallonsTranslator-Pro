# Dango-style panel transition: fade when switching stack index.
# Used for right panel (Drawing/Text/Spell check) and Comic <-> Config.

from qtpy.QtWidgets import QStackedWidget, QGraphicsOpacityEffect
from qtpy.QtCore import QPropertyAnimation, QEasingCurve


class AnimatedStackWidget(QStackedWidget):
    """QStackedWidget that fades out current page, switches, then fades in new page (Dango Phase D)."""

    def __init__(self, parent=None, duration_ms: int = 180):
        super().__init__(parent)
        self._duration_ms = duration_ms
        self._animating = False
        self._pending_index = -1
        self._anim_ref = []  # keep ref so animation is not gc'd

    def setCurrentIndex(self, index: int):
        if self._animating or index == self.currentIndex():
            super().setCurrentIndex(index)
            return
        if index < 0 or index >= self.count():
            return
        self._pending_index = index
        self._animating = True
        current = self.currentWidget()
        if current is None:
            self._do_switch()
            return
        effect = QGraphicsOpacityEffect(current)
        effect.setOpacity(1.0)
        current.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(self._duration_ms)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        self._anim_ref[:] = [anim]

        def on_fade_out_finished():
            current.setGraphicsEffect(None)
            self._anim_ref.clear()
            self._do_switch()

        anim.finished.connect(on_fade_out_finished)
        anim.start()

    def _do_switch(self):
        idx = self._pending_index
        self._pending_index = -1
        super().setCurrentIndex(idx)
        incoming = self.currentWidget()
        if incoming is None:
            self._animating = False
            return
        effect = QGraphicsOpacityEffect(incoming)
        effect.setOpacity(0.0)
        incoming.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(self._duration_ms)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        self._anim_ref[:] = [anim]

        def on_fade_in_finished():
            if incoming.graphicsEffect() == effect:
                incoming.setGraphicsEffect(None)
            self._anim_ref.clear()
            self._animating = False

        anim.finished.connect(on_fade_in_finished)
        anim.start()
