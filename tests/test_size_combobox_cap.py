"""Tests for SizeComboBox.value() clamping fix (#920)."""
import sys
import types
import os.path as osp

# Add project root so ui.* imports resolve
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

# Stub Qt before importing ui modules
qtcore = types.ModuleType("qtpy.QtCore")
class _MockQt:
    class Orientation:
        Horizontal = 1
        Vertical = 2
    class AlignmentFlag:
        AlignCenter = 1
    class PenStyle:
        SolidLine = 1
    class GlobalColor:
        transparent = 1
        black = 2
        white = 3
    class CompositionMode:
        CompositionMode_SourceOver = 1
    class RenderHint:
        SmoothPixmapTransform = 1
qtcore.Qt = _MockQt
qtcore.Signal = lambda *a, **k: None
class _MockQObject:
    def __init__(self, *a, **k): pass
qtcore.QObject = _MockQObject
qtcore.QEvent = object
qtcore.QPoint = object
qtcore.QPointF = object
qtcore.QSize = object
qtcore.QSizeF = object
qtcore.QRect = object
qtcore.QRectF = object
qtcore.QMimeData = object
qtcore.QTimer = object
qtcore.QPropertyAnimation = object
qtcore.QAbstractAnimation = object
class _MockEasingCurve:
    Linear = 0
    InOutCubic = 1
qtcore.QEasingCurve = _MockEasingCurve
qtcore.QParallelAnimationGroup = object
qtcore.QModelIndex = object
class _MockVariantAnimation:
    def __init__(self, *a, **k): pass
    def setDuration(self, v): pass
    def setStartValue(self, v): pass
    def setEndValue(self, v): pass
qtcore.QVariantAnimation = _MockVariantAnimation
qtcore.QAbstractItemModel = object
qtcore.QItemSelection = object
qtcore.QThread = object
class _MockProperty:
    def __init__(self, *a, **k):
        pass
    def setter(self, fn):
        return fn
    def __call__(self, fn):
        return self
qtcore.Property = _MockProperty
sys.modules["qtpy.QtCore"] = qtcore

class _MockNotation:
    StandardNotation = 0
class _MockDoubleValidator:
    Notation = _MockNotation
    def setTop(self, v): pass
    def setBottom(self, v): pass
    def setNotation(self, n): pass

qtgui = types.ModuleType("qtpy.QtGui")
qtgui.QDoubleValidator = _MockDoubleValidator
qtgui.QFont = object
qtgui.QColor = object
qtgui.QPen = object
qtgui.QBrush = object
qtgui.QPalette = object
qtgui.QPixmap = object
qtgui.QImage = object
qtgui.QKeyEvent = object
qtgui.QMouseEvent = object
qtgui.QCursor = object
qtgui.QKeySequence = object
qtgui.QTextCursor = object
qtgui.QTextCharFormat = object
qtgui.QTextDocument = object
qtgui.QAbstractTextDocumentLayout = object
qtgui.QTextFrameFormat = object
qtgui.QGradient = object
qtgui.QLinearGradient = object
qtgui.QRadialGradient = object
qtgui.QTransform = object
qtgui.QPolygonF = object
qtgui.QBitmap = object
qtgui.QRegion = object
qtgui.QPainterPath = object
qtgui.QPainter = object
qtgui.QInputMethodEvent = object
qtgui.QDragEnterEvent = object
qtgui.QDropEvent = object
qtgui.QDrag = object
qtgui.QIcon = object
qtgui.QFontMetrics = object
qtgui.QFontMetricsF = object
qtgui.QCloseEvent = object
qtgui.QShowEvent = object
qtgui.QHideEvent = object
qtgui.QFocusEvent = object
qtgui.QMoveEvent = object
qtgui.QResizeEvent = object
qtgui.QWheelEvent = object
qtgui.QContextMenuEvent = object
sys.modules["qtpy.QtGui"] = qtgui

qtwidgets = types.ModuleType("qtpy.QtWidgets")

class MockLayout:
    def __init__(self, *a, **k):
        pass
    def addWidget(self, *a, **k):
        pass
    def addLayout(self, *a, **k):
        pass
    def addStretch(self, *a, **k):
        pass
    def setSpacing(self, *a, **k):
        pass
    def setContentsMargins(self, *a, **k):
        pass

class MockSignal:
    def connect(self, *a, **k):
        pass
    def emit(self, *a, **k):
        pass
    def disconnect(self, *a, **k):
        pass

class MockComboBox:
    def __init__(self, *a, **k):
        self._items = []
        self._current_text = ""
        self._validator = None
        self._editable = False
        self.editTextChanged = MockSignal()
        self.activated = MockSignal()
        self.currentTextChanged = MockSignal()
    def addItems(self, items):
        self._items.extend(items)
    def setValidator(self, v):
        self._validator = v
    def setEditable(self, e):
        self._editable = e
    def currentText(self):
        return self._current_text
    def setCurrentText(self, t):
        self._current_text = t
    def hasFocus(self):
        return False
    def view(self):
        return self
    def isVisible(self):
        return False
    def setMinimumWidth(self, *a):
        pass

class _MockMsgBox(object):
    class StandardButton:
        Ok = 1
        Yes = 2
        No = 4
        Cancel = 8
_EXTRA_WIDGETS = ['QVBoxLayout', 'QHBoxLayout', 'QGridLayout', 'QFrame', 'QLabel', 'QPushButton', 'QLineEdit', 'QPlainTextEdit', 'QCheckBox', 'QSizePolicy', 'QApplication', 'QGraphicsItem', 'QGraphicsSceneHoverEvent', 'QGraphicsTextItem', 'QStyleOptionGraphicsItem', 'QStyle', 'QGraphicsSceneMouseEvent', 'QWidget', 'QMessageBox', 'QFileDialog', 'QDialog', 'QFontComboBox', 'QSlider', 'QGroupBox', 'QSpinBox', 'QDoubleSpinBox', 'QComboBox', 'QTextEdit', 'QScrollArea', 'QStackedWidget', 'QGraphicsDropShadowEffect', 'QInputDialog', 'QGraphicsItemGroup', 'QAbstractScrollArea', 'QGraphicsProxyWidget', 'QGraphicsObject', 'QGraphicsOpacityEffect', 'QGraphicsBlurEffect', 'QGraphicsColorizeEffect', 'QSpacerItem', 'QToolButton', 'QMenu', 'QAction', 'QActionGroup', 'QTreeView', 'QSplitter', 'QKeySequenceEdit', 'QTabWidget', 'QTabBar', 'QProgressBar', 'QRadioButton', 'QButtonGroup', 'QListView', 'QTableView', 'QHeaderView', 'QAbstractItemView', 'QItemSelectionModel', 'QStyledItemDelegate', 'QStyleOptionViewItem', 'QStyleOptionComboBox', 'QStyleOptionProgressBar', 'QStyleOptionButton', 'QLayout', 'QWidgetItem', 'QLayoutItem', 'QColorDialog', 'QStyleOptionSlider', 'QShortcut', 'QCompleter', 'QSystemTrayIcon']
for _name in _EXTRA_WIDGETS:
    if _name == 'QMessageBox':
        setattr(qtwidgets, _name, _MockMsgBox)
    else:
        setattr(qtwidgets, _name, MockLayout if 'Layout' in _name else object)

qtwidgets.QComboBox = MockComboBox
sys.modules["qtpy.QtWidgets"] = qtwidgets

from ui.custom_widget.combobox import SizeComboBox


def test_value_clamps_to_max():
    box = SizeComboBox([1, 300], 'font_size')
    box.setCurrentText("99999")
    assert box.value() == 300, f"Expected 300, got {box.value()}"


def test_value_clamps_to_min():
    box = SizeComboBox([10, 300], 'font_size')
    box.setCurrentText("-50")
    assert box.value() == 10, f"Expected 10, got {box.value()}"


def test_value_within_range_passes_through():
    box = SizeComboBox([1, 300], 'font_size')
    box.setCurrentText("42")
    assert box.value() == 42, f"Expected 42, got {box.value()}"


def test_setvalue_clamps():
    box = SizeComboBox([1, 300], 'font_size')
    box.setValue(500)
    assert box.value() == 300, f"Expected 300 after setValue(500), got {box.value()}"


def test_change_by_delta_respects_max():
    box = SizeComboBox([1, 300], 'font_size')
    box.setValue(250)
    box.changeByDelta(100, multiplier=1.0)
    assert box.value() == 300, f"Expected 300, got {box.value()}"
