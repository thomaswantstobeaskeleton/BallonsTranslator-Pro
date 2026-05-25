"""Tests for module_parse_widgets fallback fix (#904)."""
import sys
import types
import os.path as osp

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

class _MockSignal:
    def connect(self, *a, **k): pass
    def emit(self, *a, **k): pass

class _MockWidget:
    def __init__(self, *a, **k): pass
    def addWidget(self, *a, **k): pass
    def addLayout(self, *a, **k): pass
    def addStretch(self, *a, **k): pass
    def setSpacing(self, *a, **k): pass
    def setContentsMargins(self, *a, **k): pass
    def setColumnStretch(self, *a, **k): pass
    def setRowStretch(self, *a, **k): pass
    def setToolTip(self, *a, **k): pass
    def setText(self, *a, **k): pass
    def setChecked(self, *a, **k): pass
    def setCurrentText(self, *a, **k): pass
    def setEditable(self, *a, **k): pass
    def setValidator(self, *a, **k): pass
    def setFixedWidth(self, *a, **k): pass
    def setFixedHeight(self, *a, **k): pass
    def setMaximumWidth(self, *a, **k): pass
    def setMinimumWidth(self, *a, **k): pass
    def setMinimumHeight(self, *a, **k): pass
    def setFont(self, *a, **k): pass
    def hide(self, *a, **k): pass
    def show(self, *a, **k): pass
    def blockSignals(self, *a, **k): pass
    def setEnabled(self, *a, **k): pass
    def setStyleSheet(self, *a, **k): pass
    def setPlaceholderText(self, *a, **k): pass
    def setReadOnly(self, *a, **k): pass
    def setFocus(self, *a, **k): pass
    def setPlainText(self, *a, **k): pass
    def setAlignment(self, *a, **k): pass
    def setWordWrap(self, *a, **k): pass
    def text(self, *a, **k): return ""
    def isChecked(self, *a, **k): return False
    def value(self, *a, **k): return 0
    def currentText(self, *a, **k): return ""
    def count(self, *a, **k): return 0
    def isVisible(self, *a, **k): return False
    def mapToGlobal(self, *a, **k): pass
    def view(self, *a, **k): return self
    def setItemDelegate(self, *a, **k): pass
    def setMouseTracking(self, *a, **k): pass
    def setCursor(self, *a, **k): pass
    def viewport(self, *a, **k): return self
    def update(self, *a, **k): pass
    def parent(self, *a, **k): return None
    def setParent(self, *a, **k): pass
    def size(self, *a, **k): return self
    def width(self, *a, **k): return 100
    def height(self, *a, **k): return 30
    def pos(self, *a, **k): return self
    def x(self, *a, **k): return 0
    def y(self, *a, **k): return 0
    def rect(self, *a, **k): return self
    def geometry(self, *a, **k): return self
    def frameGeometry(self, *a, **k): return self
    def setGeometry(self, *a, **k): pass
    def move(self, *a, **k): pass
    def resize(self, *a, **k): pass
    def close(self, *a, **k): pass
    def deleteLater(self, *a, **k): pass
    def raise_(self, *a, **k): pass
    def lower(self, *a, **k): pass
    def stackUnder(self, *a, **k): pass
    def activateWindow(self, *a, **k): pass
    def setWindowTitle(self, *a, **k): pass
    def setWindowIcon(self, *a, **k): pass
    def setWindowFlags(self, *a, **k): pass
    def windowFlags(self, *a, **k): return 0
    def isActiveWindow(self, *a, **k): return False
    def hasFocus(self, *a, **k): return False
    def setFocusPolicy(self, *a, **k): pass
    def setAttribute(self, *a, **k): pass
    def testAttribute(self, *a, **k): return False
    def findChild(self, *a, **k): return None
    def findChildren(self, *a, **k): return []
    def children(self, *a, **k): return []
    def actions(self, *a, **k): return []
    def addAction(self, *a, **k): pass
    def removeAction(self, *a, **k): pass
    def clear(self, *a, **k): pass
    def setSizePolicy(self, *a, **k): pass
    def sizePolicy(self, *a, **k): return self
    def setAcceptDrops(self, *a, **k): pass
    def acceptDrops(self, *a, **k): return False
    def setDragEnabled(self, *a, **k): pass
    def setDropIndicatorShown(self, *a, **k): pass
    def setDefaultDropAction(self, *a, **k): pass
    def setDragDropMode(self, *a, **k): pass
    def setDragDropOverwriteMode(self, *a, **k): pass
    def setSelectionMode(self, *a, **k): pass
    def setSelectionBehavior(self, *a, **k): pass
    def setAlternatingRowColors(self, *a, **k): pass
    def setUniformRowHeights(self, *a, **k): pass
    def setAnimated(self, *a, **k): pass
    def setAllColumnsShowFocus(self, *a, **k): pass
    def setAutoExpandDelay(self, *a, **k): pass
    def setIndentation(self, *a, **k): pass
    def setRootIsDecorated(self, *a, **k): pass
    def setSortingEnabled(self, *a, **k): pass
    def sortByColumn(self, *a, **k): pass
    def header(self, *a, **k): return self
    def setHeaderHidden(self, *a, **k): pass
    def setColumnWidth(self, *a, **k): pass
    def columnWidth(self, *a, **k): return 100
    def setColumnCount(self, *a, **k): pass
    def columnCount(self, *a, **k): return 1
    def setRowCount(self, *a, **k): pass
    def rowCount(self, *a, **k): return 0
    def setCellWidget(self, *a, **k): pass
    def cellWidget(self, *a, **k): return None
    def item(self, *a, **k): return None
    def setItem(self, *a, **k): pass
    def takeItem(self, *a, **k): return None
    def addItem(self, *a, **k): pass
    def insertItem(self, *a, **k): pass
    def removeItem(self, *a, **k): pass
    def itemText(self, *a, **k): return ""
    def itemData(self, *a, **k): return None
    def setItemData(self, *a, **k): pass
    def setItemText(self, *a, **k): pass
    def setIconSize(self, *a, **k): pass
    def iconSize(self, *a, **k): return self
    def setUniformItemSizes(self, *a, **k): pass
    def setResizeMode(self, *a, **k): pass
    def setViewMode(self, *a, **k): pass
    def setFlow(self, *a, **k): pass
    def setWrapping(self, *a, **k): pass
    def setSpacing(self, *a, **k): pass
    def setGridSize(self, *a, **k): pass
    def setBatchSize(self, *a, **k): pass
    def setLayoutMode(self, *a, **k): pass
    def setMovement(self, *a, **k): pass
    def setWordWrap(self, *a, **k): pass
    def setTextElideMode(self, *a, **k): pass
    def setHorizontalScrollMode(self, *a, **k): pass
    def setVerticalScrollMode(self, *a, **k): pass
    def setTabPosition(self, *a, **k): pass
    def setTabShape(self, *a, **k): pass
    def setDocumentMode(self, *a, **k): pass
    def setMovable(self, *a, **k): pass
    def setTabsClosable(self, *a, **k): pass
    def setTabBarAutoHide(self, *a, **k): pass
    def setUsesScrollButtons(self, *a, **k): pass
    def setCornerWidget(self, *a, **k): pass
    def setOrientation(self, *a, **k): pass
    def setOpaqueResize(self, *a, **k): pass
    def setHandleWidth(self, *a, **k): pass
    def setStretchFactor(self, *a, **k): pass
    def setSizes(self, *a, **k): pass
    def sizes(self, *a, **k): return []
    def addWidget(self, *a, **k): pass
    def insertWidget(self, *a, **k): pass
    def widget(self, *a, **k): return None
    def indexOf(self, *a, **k): return -1
    def setCurrentIndex(self, *a, **k): pass
    def currentIndex(self, *a, **k): return 0
    def setCurrentWidget(self, *a, **k): pass
    def removeWidget(self, *a, **k): pass
    def setLineWidth(self, *a, **k): pass
    def setMidLineWidth(self, *a, **k): pass
    def setFrameStyle(self, *a, **k): pass
    def setFrameShape(self, *a, **k): pass
    def setFrameShadow(self, *a, **k): pass
    def setAutoFillBackground(self, *a, **k): pass
    def setBackgroundRole(self, *a, **k): pass
    def setForegroundRole(self, *a, **k): pass
    def setContentsMargins(self, *a, **k): pass
    def getContentsMargins(self, *a, **k): return (0, 0, 0, 0)
    def contentsMargins(self, *a, **k): return self
    def contentsRect(self, *a, **k): return self
    def setSpacing(self, *a, **k): pass
    def spacing(self, *a, **k): return 0
    def setStretch(self, *a, **k): pass
    def stretch(self, *a, **k): return 0
    def setAlignment(self, *a, **k): pass
    def alignment(self, *a, **k): return 0
    def setMenu(self, *a, **k): pass
    def menu(self, *a, **k): return None
    def setPopupMode(self, *a, **k): pass
    def setToolButtonStyle(self, *a, **k): pass
    def setArrowType(self, *a, **k): pass
    def setDefaultAction(self, *a, **k): pass
    def setCheckable(self, *a, **k): pass
    def setChecked(self, *a, **k): pass
    def isChecked(self, *a, **k): return False
    def setAutoRaise(self, *a, **k): pass
    def setDown(self, *a, **k): pass
    def isDown(self, *a, **k): return False
    def animateClick(self, *a, **k): pass
    def click(self, *a, **k): pass
    def toggle(self, *a, **k): pass
    def setShortcut(self, *a, **k): pass
    def setShortcuts(self, *a, **k): pass
    def setAutoRepeat(self, *a, **k): pass
    def setAutoRepeatDelay(self, *a, **k): pass
    def setAutoRepeatInterval(self, *a, **k): pass
    def setText(self, *a, **k): pass
    def text(self, *a, **k): return ""
    def setIcon(self, *a, **k): pass
    def icon(self, *a, **k): return None
    def setWhatsThis(self, *a, **k): pass
    def setStatusTip(self, *a, **k): pass
    def setShortcutContext(self, *a, **k): pass
    def setData(self, *a, **k): pass
    def data(self, *a, **k): return None
    def setFlags(self, *a, **k): pass
    def flags(self, *a, **k): return 0
    def setCheckState(self, *a, **k): pass
    def checkState(self, *a, **k): return 0
    def setTristate(self, *a, **k): pass
    def isTristate(self, *a, **k): return False
    def setExclusive(self, *a, **k): pass
    def setId(self, *a, **k): pass
    def id(self, *a, **k): return -1
    def setSingleStep(self, *a, **k): pass
    def setPageStep(self, *a, **k): pass
    def setRange(self, *a, **k): pass
    def setSliderPosition(self, *a, **k): pass
    def sliderPosition(self, *a, **k): return 0
    def setInvertedAppearance(self, *a, **k): pass
    def setInvertedControls(self, *a, **k): pass
    def setTracking(self, *a, **k): pass
    def setTickPosition(self, *a, **k): pass
    def setTickInterval(self, *a, **k): pass
    def setWrapping(self, *a, **k): pass
    def setSpecialValueText(self, *a, **k): pass
    def setPrefix(self, *a, **k): pass
    def setSuffix(self, *a, **k): pass
    def setDecimals(self, *a, **k): pass
    def setMaximum(self, *a, **k): pass
    def setMinimum(self, *a, **k): pass
    def maximum(self, *a, **k): return 9999
    def minimum(self, *a, **k): return 0
    def setSingleStep(self, *a, **k): pass
    def setDisplayIntegerBase(self, *a, **k): pass
    def setDisplayAlignment(self, *a, **k): pass
    def setButtonSymbols(self, *a, **k): pass
    def setCorrectionMode(self, *a, **k): pass
    def setKeyboardTracking(self, *a, **k): pass
    def setAccelerated(self, *a, **k): pass
    def selectAll(self, *a, **k): pass
    def deselect(self, *a, **k): pass
    def insert(self, *a, **k): pass
    def undo(self, *a, **k): pass
    def redo(self, *a, **k): pass
    def cut(self, *a, **k): pass
    def copy(self, *a, **k): pass
    def paste(self, *a, **k): pass
    def clear(self, *a, **k): pass
    def selectAll(self, *a, **k): pass
    def home(self, *a, **k): pass
    def end(self, *a, **k): pass
    def setEchoMode(self, *a, **k): pass
    def setInputMask(self, *a, **k): pass
    def setMaxLength(self, *a, **k): pass
    def maxLength(self, *a, **k): return 32767
    def setCursorPosition(self, *a, **k): pass
    def cursorPosition(self, *a, **k): return 0
    def setTabChangesFocus(self, *a, **k): pass
    def setAcceptRichText(self, *a, **k): pass
    def setOverwriteMode(self, *a, **k): pass
    def setTextInteractionFlags(self, *a, **k): pass
    def setVisible(self, *a, **k): pass
    def isVisible(self, *a, **k): return False
    def setEnabled(self, *a, **k): pass
    def isEnabled(self, *a, **k): return True
    def setHidden(self, *a, **k): pass
    def setLayout(self, *a, **k): pass
    def layout(self, *a, **k): return None
    def font(self, *a, **k): return _MockFont()
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        if 'changed' in name.lower() or 'clicked' in name.lower() or 'pressed' in name.lower() or 'triggered' in name.lower() or 'editingFinished' in name or 'activated' in name or 'textChanged' in name or 'stateChanged' in name or 'currentIndexChanged' in name or 'focus_out' in name or 'focus_in' in name or name == 'entered' or name == 'leaved' or name == 'hoverEnter' or name == 'hoverLeave':
            return _MockSignal()
        return _MockWidget() if name[0].isupper() else lambda *a, **k: None

class _MockVariantAnimation(_MockWidget):
    def setDuration(self, v): pass
    def setStartValue(self, v): pass
    def setEndValue(self, v): pass
    valueChanged = _MockSignal()

class _MockMsgBox(object):
    class StandardButton:
        Ok = 1; Yes = 2; No = 4; Cancel = 8

# Minimal Qt stubs for module_parse_widgets
qtcore = types.ModuleType("qtpy.QtCore")
class _MockQt:
    class Orientation:
        Horizontal = 1
        Vertical = 2
    class AlignmentFlag:
        AlignCenter = 1; AlignLeft = 2; AlignRight = 4; AlignTop = 8; AlignBottom = 16; AlignVCenter = 32; AlignHCenter = 64
    class PenStyle:
        SolidLine = 1
    class GlobalColor:
        transparent = 1; black = 2; white = 3
    class CompositionMode:
        CompositionMode_SourceOver = 1
    class RenderHint:
        SmoothPixmapTransform = 1
    class WidgetAttribute:
        WA_TransparentForMouseEvents = 1
        WA_MouseTracking = 2
        WA_TranslucentBackground = 3
        WA_NoSystemBackground = 4
        WA_OpaquePaintEvent = 5
qtcore.Qt = _MockQt
def _make_signal(*a, **k):
    return _MockSignal()
qtcore.Signal = _make_signal
class _MockQObject:
    def __init__(self, *a, **k): pass
    def installEventFilter(self, *a, **k): pass
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
qtcore.QParallelAnimationGroup = object
qtcore.QModelIndex = object
qtcore.QVariantAnimation = _MockVariantAnimation
qtcore.QAbstractItemModel = object
qtcore.QItemSelection = object
qtcore.QThread = object
class _MockEasingCurve:
    Linear = 0; InOutCubic = 1
qtcore.QEasingCurve = _MockEasingCurve
class _MockProperty:
    def __init__(self, *a, **k): pass
    def setter(self, fn): return fn
    def __call__(self, fn): return self
qtcore.Property = _MockProperty
sys.modules["qtpy.QtCore"] = qtcore

qtgui = types.ModuleType("qtpy.QtGui")
class _MockDoubleValidator:
    class Notation:
        StandardNotation = 0
    def setTop(self, v): pass
    def setBottom(self, v): pass
    def setNotation(self, n): pass
qtgui.QDoubleValidator = _MockDoubleValidator
class _MockFont:
    def setPointSizeF(self, *a, **k): pass
    def setPointSize(self, *a, **k): pass
    def setWeight(self, *a, **k): pass
    def setFamily(self, *a, **k): pass
    def setBold(self, *a, **k): pass
    def setItalic(self, *a, **k): pass
    def setUnderline(self, *a, **k): pass
    def setStrikeOut(self, *a, **k): pass
    def setStyleStrategy(self, *a, **k): pass
    def pointSizeF(self, *a, **k): return 12.0
qtgui.QFont = _MockFont
class _MockColor:
    def __init__(self, *a, **k): pass
qtgui.QColor = _MockColor
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
for _name in ['QWidget', 'QHBoxLayout', 'QVBoxLayout', 'QGridLayout', 'QLabel', 'QLineEdit', 'QPlainTextEdit', 'QCheckBox', 'QComboBox', 'QPushButton', 'QSizePolicy', 'QMessageBox', 'QFileDialog', 'QDialog', 'QScrollArea', 'QApplication', 'QAbstractScrollArea', 'QGraphicsItem', 'QGraphicsSceneHoverEvent', 'QGraphicsTextItem', 'QStyleOptionGraphicsItem', 'QStyle', 'QGraphicsSceneMouseEvent', 'QFrame', 'QSlider', 'QGroupBox', 'QSpinBox', 'QDoubleSpinBox', 'QTextEdit', 'QStackedWidget', 'QGraphicsDropShadowEffect', 'QInputDialog', 'QGraphicsItemGroup', 'QGraphicsProxyWidget', 'QGraphicsObject', 'QGraphicsOpacityEffect', 'QGraphicsBlurEffect', 'QGraphicsColorizeEffect', 'QSpacerItem', 'QToolButton', 'QMenu', 'QAction', 'QActionGroup', 'QTreeView', 'QSplitter', 'QKeySequenceEdit', 'QTabWidget', 'QTabBar', 'QProgressBar', 'QRadioButton', 'QButtonGroup', 'QListView', 'QTableView', 'QHeaderView', 'QAbstractItemView', 'QItemSelectionModel', 'QStyledItemDelegate', 'QStyleOptionViewItem', 'QStyleOptionComboBox', 'QStyleOptionProgressBar', 'QStyleOptionButton', 'QLayout', 'QWidgetItem', 'QLayoutItem', 'QColorDialog', 'QStyleOptionSlider', 'QShortcut', 'QCompleter', 'QSystemTrayIcon']:
    if _name == 'QMessageBox':
        setattr(qtwidgets, _name, _MockMsgBox)
    else:
        setattr(qtwidgets, _name, _MockWidget)
sys.modules["qtpy.QtWidgets"] = qtwidgets

from ui.module_parse_widgets import ParamWidget


def test_unknown_param_type_falls_back_to_line_editor():
    """A param dict with an unrecognised 'type' should not crash; it should fall back to a line editor."""
    params = {
        'weird_param': {
            'type': 'unknown_future_type',
            'value': 'fallback_test',
            'description': 'A param with a type the UI does not yet know about.'
        }
    }
    # Should not raise ValueError anymore (fix #904)
    w = ParamWidget(params)
    assert w is not None


def test_selector_with_missing_options_falls_back():
    """A selector param whose 'options' list is missing should not crash."""
    params = {
        'model path': {
            'type': 'selector',
            # Deliberately no 'options' key to simulate fresh install with empty CKPT_LIST
            'value': 'data/models/default.pt',
            'display_name': 'Model path'
        }
    }
    w = ParamWidget(params)
    assert w is not None
