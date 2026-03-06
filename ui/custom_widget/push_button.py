from qtpy.QtWidgets import QPushButton

from .hover_animation import install_hover_opacity_animation, install_hover_scale_animation


class NoBorderPushBtn(QPushButton):
    """Push button with bubbly hover feedback (opacity + slight scale)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_hover_opacity_animation(self, duration_ms=100, normal_opacity=0.88)
        install_hover_scale_animation(self, duration_ms=80, size_delta=(3, 2))