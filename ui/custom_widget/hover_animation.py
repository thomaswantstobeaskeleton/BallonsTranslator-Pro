# Dango-style hover feedback: subtle opacity animation on enter/leave.
# Uses QGraphicsOpacityEffect + QPropertyAnimation for smooth "bounce" feel without layout shifts.
# install_hover_scale_animation: optional size pulse (1.0 -> ~1.05) for a more pronounced bounce (Dango Phase C).

from qtpy.QtWidgets import QWidget, QGraphicsOpacityEffect
from qtpy.QtCore import QPropertyAnimation, QObject, QEvent, QEasingCurve, QSize


def install_hover_opacity_animation(widget: QWidget, duration_ms: int = 100, hover_opacity: float = 1.0, normal_opacity: float = 0.88):
    """
    Install a hover opacity animation on a widget (Dango-style soft feedback).
    On enter: opacity animates from normal_opacity to hover_opacity.
    On leave: opacity animates back to normal_opacity.
    """
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(normal_opacity)
    widget.setGraphicsEffect(effect)

    anim = QPropertyAnimation(effect, b"opacity")
    anim.setDuration(duration_ms)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.setParent(widget)

    def on_enter(_e):
        anim.stop()
        anim.setStartValue(effect.opacity())
        anim.setEndValue(hover_opacity)
        anim.start()

    def on_leave(_e):
        anim.stop()
        anim.setStartValue(effect.opacity())
        anim.setEndValue(normal_opacity)
        anim.start()

    filt = _HoverOpacityFilter(widget, on_enter, on_leave)
    widget.installEventFilter(filt)


def install_hover_scale_animation(widget: QWidget, duration_ms: int = 80, size_delta: tuple = (4, 2)):
    """
    Dango Phase C: subtle size "bounce" on hover.
    Animates minimumSize and maximumSize so the widget grows slightly (e.g. +4px width, +2px height).
    Base (rest) size is captured once on first hover; we always animate back to that base on leave.
    """
    base_size = [None, None]  # [width, height] — rest size, set once on first enter

    anim_min = QPropertyAnimation(widget, b"minimumSize")
    anim_min.setDuration(duration_ms)
    anim_min.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim_min.setParent(widget)
    anim_max = QPropertyAnimation(widget, b"maximumSize")
    anim_max.setDuration(duration_ms)
    anim_max.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim_max.setParent(widget)

    def on_enter(_e):
        # Capture rest size only once when we have valid size (first time from rest state)
        if base_size[0] is None or base_size[1] is None:
            w, h = widget.size().width(), widget.size().height()
            if w > 0 and h > 0:
                base_size[0], base_size[1] = w, h
            else:
                return
        bw, bh = base_size[0], base_size[1]
        large_w = bw + size_delta[0]
        large_h = bh + size_delta[1]
        # Animate from current size to large (so we don't jump if already mid-animation)
        cur_min = widget.minimumSize()
        cur_max = widget.maximumSize()
        anim_min.stop()
        anim_max.stop()
        anim_min.setStartValue(cur_min)
        anim_min.setEndValue(QSize(large_w, large_h))
        anim_max.setStartValue(cur_max)
        anim_max.setEndValue(QSize(large_w, large_h))
        anim_min.start()
        anim_max.start()

    def on_leave(_e):
        if base_size[0] is None or base_size[1] is None:
            return
        bw, bh = base_size[0], base_size[1]
        # Always animate back to stored rest size (never read widget.size() here — it may already be large)
        cur_min = widget.minimumSize()
        cur_max = widget.maximumSize()
        anim_min.stop()
        anim_max.stop()
        anim_min.setStartValue(cur_min)
        anim_min.setEndValue(QSize(bw, bh))
        anim_max.setStartValue(cur_max)
        anim_max.setEndValue(QSize(bw, bh))
        anim_min.start()
        anim_max.start()

    filt = _HoverOpacityFilter(widget, on_enter, on_leave)
    widget.installEventFilter(filt)


class _HoverOpacityFilter(QObject):
    """Event filter to trigger enter/leave animations."""
    def __init__(self, target, on_enter, on_leave):
        super().__init__(target)
        self._target = target
        self._on_enter = on_enter
        self._on_leave = on_leave

    def eventFilter(self, obj, event):
        if obj != self._target:
            return False
        t = event.type()
        if t == QEvent.Type.Enter:
            self._on_enter(event)
        elif t == QEvent.Type.Leave:
            self._on_leave(event)
        return False
