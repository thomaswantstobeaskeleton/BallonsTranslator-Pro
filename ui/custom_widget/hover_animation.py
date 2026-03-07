# Dango-style hover and click feedback: opacity (and optional scale) animation.
# Uses QGraphicsOpacityEffect + QPropertyAnimation for smooth feedback without layout shifts.
# install_hover_scale_animation: optional size pulse on hover (Dango Phase C).
# Optional press_opacity: brief opacity dip on mouse press for click feedback.
# When pcfg.reduce_motion is True, durations are 0 and scale animation is skipped.

from qtpy.QtWidgets import QWidget, QGraphicsOpacityEffect
from qtpy.QtCore import QPropertyAnimation, QObject, QEvent, QEasingCurve, QSize, Qt
from qtpy.QtGui import QMouseEvent

try:
    from utils.config import pcfg
except Exception:
    pcfg = None


def _effective_duration_ms(requested_ms: int) -> int:
    """Return 0 when reduce_motion is on, else requested_ms."""
    if pcfg is None:
        return requested_ms
    return 0 if getattr(pcfg, 'reduce_motion', False) else requested_ms


def _reduce_motion_skip_scale() -> bool:
    """True when scale animation should be skipped (reduce_motion)."""
    if pcfg is None:
        return False
    return getattr(pcfg, 'reduce_motion', False)


def install_hover_opacity_animation(
    widget: QWidget,
    duration_ms: int = 100,
    hover_opacity: float = 1.0,
    normal_opacity: float = 0.88,
    press_opacity: float = None,
    press_duration_ms: int = 70,
):
    """
    Install hover (and optional press) opacity animation on a widget.
    Enter: opacity -> hover_opacity. Leave: opacity -> normal_opacity.
    If press_opacity is set (e.g. 0.72): on left-button press opacity -> press_opacity;
    on release -> hover_opacity if still under mouse else normal_opacity.
    """
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(normal_opacity)
    widget.setGraphicsEffect(effect)

    anim = QPropertyAnimation(effect, b"opacity")
    anim.setDuration(duration_ms)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.setParent(widget)

    press_anim_duration = press_duration_ms if press_opacity is not None else duration_ms

    def go_to(value: float, duration: int = duration_ms):
        anim.stop()
        anim.setDuration(_effective_duration_ms(duration))
        anim.setStartValue(effect.opacity())
        anim.setEndValue(value)
        anim.start()

    def on_enter(_e):
        go_to(hover_opacity)

    def on_leave(_e):
        go_to(normal_opacity)

    def on_press(e: QMouseEvent):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        go_to(press_opacity, press_anim_duration)

    def on_release(e: QMouseEvent):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        target = hover_opacity if widget.underMouse() else normal_opacity
        go_to(target, duration_ms)

    filt = _HoverPressFilter(widget, on_enter, on_leave, on_press if press_opacity is not None else None, on_release if press_opacity is not None else None)
    widget.installEventFilter(filt)


def install_button_animations(
    widget: QWidget,
    hover_opacity: float = 1.0,
    normal_opacity: float = 0.88,
    press_opacity: float = 0.72,
    duration_ms: int = 100,
    with_scale: bool = True,
    scale_delta: tuple = (3, 2),
):
    """
    One-call setup: hover + press opacity and optional hover scale (for buttons).
    Use for Run, Open, dialog actions, etc.
    When reduce_motion is True, scale animation no-ops at runtime and opacity uses 0 duration.
    """
    install_hover_opacity_animation(
        widget,
        duration_ms=duration_ms,
        hover_opacity=hover_opacity,
        normal_opacity=normal_opacity,
        press_opacity=press_opacity,
        press_duration_ms=70,
    )
    if with_scale:
        install_hover_scale_animation(widget, duration_ms=80, size_delta=scale_delta)


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

    def effective_duration():
        return _effective_duration_ms(duration_ms)

    def on_enter(_e):
        if _reduce_motion_skip_scale():
            return
        d = effective_duration()
        anim_min.setDuration(d)
        anim_max.setDuration(d)
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
        if _reduce_motion_skip_scale():
            return
        d = effective_duration()
        anim_min.setDuration(d)
        anim_max.setDuration(d)
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


class _HoverPressFilter(QObject):
    """Event filter for hover + optional press/release opacity."""
    def __init__(self, target, on_enter, on_leave, on_press=None, on_release=None):
        super().__init__(target)
        self._target = target
        self._on_enter = on_enter
        self._on_leave = on_leave
        self._on_press = on_press
        self._on_release = on_release

    def eventFilter(self, obj, event):
        if obj != self._target:
            return False
        t = event.type()
        if t == QEvent.Type.Enter:
            self._on_enter(event)
        elif t == QEvent.Type.Leave:
            self._on_leave(event)
        elif t == QEvent.Type.MouseButtonPress and self._on_press:
            self._on_press(event)
        elif t == QEvent.Type.MouseButtonRelease and self._on_release:
            self._on_release(event)
        return False
