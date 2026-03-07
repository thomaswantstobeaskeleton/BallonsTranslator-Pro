# Smooth scroll: animate scroll position on wheel for an enhanced feel.
# When smooth_scroll_duration_ms > 0, wheel events drive an animation to the target value instead of instant jump.
# Optional motion blur: brief viewport opacity dip during scroll for a latency / enhanced feel.
# When pcfg.reduce_motion is True, smooth scroll and motion blur are effectively off.

from qtpy.QtWidgets import QScrollArea, QGraphicsOpacityEffect
from qtpy.QtCore import QPropertyAnimation, QEasingCurve, QTimer

try:
    from utils.config import pcfg
except Exception:
    pcfg = None


class SmoothScrollArea(QScrollArea):
    """
    QScrollArea that animates vertical scroll on wheel.
    Set animation duration via setSmoothScrollDuration(ms); 0 disables smooth scroll.
    Set motion blur via setMotionBlurOnScroll(True) for a brief opacity dip during scroll (enhanced feel).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._smooth_duration_ms = 0
        self._anim_v = None
        self._motion_blur_on_scroll = False
        self._scroll_opacity_effect = None

    def setSmoothScrollDuration(self, ms: int):
        """Set smooth scroll animation duration in ms. 0 = off (instant scroll)."""
        self._smooth_duration_ms = max(0, int(ms))

    def smoothScrollDuration(self) -> int:
        return self._smooth_duration_ms

    def setMotionBlurOnScroll(self, on: bool):
        """When True, briefly reduce viewport opacity during scroll for a motion-blur / latency feel."""
        self._motion_blur_on_scroll = bool(on)

    def motionBlurOnScroll(self) -> bool:
        return self._motion_blur_on_scroll

    def _apply_scroll_latency_effect(self, apply: bool):
        """Apply or remove brief opacity dip on viewport during scroll (motion-blur feel).
        Deferred to next event loop to avoid 'paint device can only be painted by one painter at a time'.
        """
        if getattr(pcfg, 'reduce_motion', False) if pcfg else False:
            return
        if not self._motion_blur_on_scroll:
            return
        vp = self.viewport()
        if vp is None:
            return

        def do_apply():
            v = self.viewport()
            if v is None:
                return
            if getattr(pcfg, 'reduce_motion', False) if pcfg else False:
                return
            if not getattr(self, '_motion_blur_on_scroll', False):
                return
            if apply:
                if self._scroll_opacity_effect is None:
                    self._scroll_opacity_effect = QGraphicsOpacityEffect(v)
                    v.setGraphicsEffect(self._scroll_opacity_effect)
                self._scroll_opacity_effect.setOpacity(0.92)
            else:
                if self._scroll_opacity_effect is not None:
                    self._scroll_opacity_effect.setOpacity(1.0)

        QTimer.singleShot(0, do_apply)

    def wheelEvent(self, event):
        if getattr(pcfg, 'reduce_motion', False) if pcfg else False:
            return super().wheelEvent(event)
        if self._smooth_duration_ms <= 0:
            return super().wheelEvent(event)
        sb_v = self.verticalScrollBar()
        delta = event.angleDelta().y()
        if delta == 0:
            return super().wheelEvent(event)
        if sb_v.isVisible():
            step = (sb_v.pageStep() // 2) if abs(delta) > 120 else sb_v.singleStep() * (2 if abs(delta) > 60 else 1)
            target_v = sb_v.value() - delta // 120 * step
            target_v = max(sb_v.minimum(), min(sb_v.maximum(), target_v))
            if self._anim_v is None:
                self._anim_v = QPropertyAnimation(sb_v, b"value")
                self._anim_v.setEasingCurve(QEasingCurve.Type.OutCubic)
                self._anim_v.setParent(self)
                self._anim_v.finished.connect(self._on_smooth_scroll_finished)
            self._apply_scroll_latency_effect(True)
            self._anim_v.stop()
            self._anim_v.setDuration(self._smooth_duration_ms)
            self._anim_v.setStartValue(sb_v.value())
            self._anim_v.setEndValue(target_v)
            self._anim_v.start()
            event.accept()
            return
        return super().wheelEvent(event)

    def _on_smooth_scroll_finished(self):
        self._apply_scroll_latency_effect(False)
