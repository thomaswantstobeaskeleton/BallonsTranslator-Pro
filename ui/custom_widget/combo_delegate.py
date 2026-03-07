# Combo box list item hover animation (smooth background color transition).
# Uses QStyledItemDelegate + QVariantAnimation; apply via install_combo_hover_animation(combo).

from qtpy.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem, QApplication
from qtpy.QtCore import QModelIndex, QVariantAnimation, Qt, QAbstractItemModel, QObject, QEvent
from qtpy.QtGui import QPainter, QColor


class _ComboHoverDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover_index = -1
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(180)
        self._anim.setStartValue(QColor(0, 0, 0, 0))
        self._anim.setEndValue(QColor(0, 0, 0, 0))
        self._current_bg = QColor(0, 0, 0, 0)
        self._anim.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self, value):
        self._current_bg = value
        if self.parent() and hasattr(self.parent(), 'viewport'):
            self.parent().viewport().update()

    def set_hover_index(self, index: int):
        if index == self._hover_index:
            return
        prev = self._hover_index
        self._hover_index = index
        view = self.parent()
        if view is None:
            return
        # Animate from current color to target (hover or transparent)
        self._anim.stop()
        self._anim.setStartValue(self._current_bg)
        target = QColor(240, 240, 240, 255) if index >= 0 else QColor(0, 0, 0, 0)
        if hasattr(QApplication, 'palette') and view:
            pal = view.palette()
            if pal.window().color().value() < 128:
                target = QColor(60, 60, 60, 255) if index >= 0 else QColor(0, 0, 0, 0)
        self._anim.setEndValue(target)
        self._anim.start()

    def paint(self, painter, option, index):
        option_copy = QStyleOptionViewItem(option)
        if index.row() == self._hover_index and self._current_bg.alpha() > 0:
            painter.fillRect(option.rect, self._current_bg)
        super().paint(painter, option_copy, index)


def install_combo_hover_animation(combo):
    """
    Install a delegate on combo.view() that animates list item background on hover.
    Call after combo is constructed (e.g. in ComboBox.__init__).
    """
    view = combo.view()
    if view is None:
        return
    delegate = _ComboHoverDelegate(view)
    view.setItemDelegate(delegate)
    view.setMouseTracking(True)

    def on_entered(index):
        delegate.set_hover_index(index.row() if index.isValid() else -1)

    def on_leave(_):
        delegate.set_hover_index(-1)

    view.entered.connect(on_entered)
    leave_filter = _LeaveFilter(view, on_leave)
    view.viewport().installEventFilter(leave_filter)


class _LeaveFilter(QObject):
    """Event filter to clear hover when mouse leaves the combo viewport. Must be a QObject for installEventFilter."""
    def __init__(self, view, on_leave, parent=None):
        super().__init__(parent if parent is not None else view)
        self._view = view
        self._on_leave = on_leave

    def eventFilter(self, obj, event):
        if obj is self._view.viewport() and event.type() == QEvent.Type.Leave:
            self._on_leave(event)
        return False
