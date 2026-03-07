from qtpy.QtWidgets import QPushButton

from .hover_animation import install_button_animations


class NoBorderPushBtn(QPushButton):
    """Push button with hover + click animations (opacity and optional scale)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_button_animations(self, normal_opacity=0.88, press_opacity=0.72)