from qtpy.QtWidgets import QWidget, QFrame
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont


def _sanitize_font(font: QFont) -> QFont:
    """Ensure point size is > 0 to avoid QFont::setPointSize warnings (e.g. Windows default -1)."""
    if font.pointSizeF() <= 0 or font.pointSize() <= 0:
        f = QFont(font)
        f.setPointSizeF(10.0)
        f.setPointSize(10)  # ensure integer point size is valid (Qt may use setPointSize internally)
        return f
    return font


def sanitize_font(font: QFont) -> QFont:
    """Public helper: ensure point size is > 0. Use when setting fonts on widgets that may inherit -1."""
    return _sanitize_font(font)


class Widget(QWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def showEvent(self, event):
        # Avoid QFont::setPointSize: Point size <= 0 (-1) spam from inherited/default fonts.
        font = self.font()
        if font.pointSizeF() <= 0:
            self.setFont(_sanitize_font(font))
        for child in self.findChildren(QWidget):
            cf = child.font()
            if cf.pointSizeF() <= 0:
                child.setFont(_sanitize_font(cf))
        super().showEvent(event)


class SeparatorWidget(QFrame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)