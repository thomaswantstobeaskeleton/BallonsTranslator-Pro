import importlib
import sys
import types


def _install_dialog_stubs():
    class DummyDialog:
        def __init__(self, *_, **__):
            self._accepted = False

        def tr(self, text):
            return text

        def setWindowTitle(self, *_):
            return None

        def setMinimumSize(self, *_):
            return None

        def resize(self, *_):
            return None

        def accept(self):
            self._accepted = True

    class DummyLayout:
        def __init__(self, *_):
            self.widgets = []

        def addWidget(self, widget):
            self.widgets.append(widget)

        def addLayout(self, layout):
            self.widgets.append(layout)

        def addStretch(self):
            return None

    class DummyLabel:
        def __init__(self, *_):
            return None

        def setWordWrap(self, *_):
            return None

    class DummyButton:
        def __init__(self, *_):
            self.clicked = types.SimpleNamespace(connect=lambda *_: None)

        def setToolTip(self, *_):
            return None

        def setDefault(self, *_):
            return None

        def setFocus(self):
            return None

    class DummyCheckBox:
        def __init__(self, *_):
            self._checked = False
            self.props = {}

        def setToolTip(self, *_):
            return None

        def setProperty(self, key, value):
            self.props[key] = value

        def setChecked(self, value):
            self._checked = bool(value)

        def isChecked(self):
            return self._checked

    class DummyMessageBox:
        class StandardButton:
            Yes = 1
            No = 2

        warning_result = StandardButton.Yes

        @classmethod
        def warning(cls, *_, **__):
            return cls.warning_result

    class DummyGroupBox:
        def __init__(self, *_):
            return None

    qtwidgets = types.ModuleType("qtpy.QtWidgets")
    qtwidgets.QDialog = DummyDialog
    qtwidgets.QVBoxLayout = DummyLayout
    qtwidgets.QHBoxLayout = DummyLayout
    qtwidgets.QLabel = DummyLabel
    qtwidgets.QPushButton = DummyButton
    qtwidgets.QCheckBox = DummyCheckBox
    qtwidgets.QScrollArea = DummyGroupBox
    qtwidgets.QWidget = DummyGroupBox
    qtwidgets.QGroupBox = DummyGroupBox
    qtwidgets.QFrame = DummyGroupBox
    qtwidgets.QMessageBox = DummyMessageBox

    qtcore = types.ModuleType("qtpy.QtCore")
    qtcore.Qt = object
    qtcore.Signal = lambda *_, **__: None

    qtpy = types.ModuleType("qtpy")
    sys.modules["qtpy"] = qtpy
    sys.modules["qtpy.QtWidgets"] = qtwidgets
    sys.modules["qtpy.QtCore"] = qtcore

    packages = types.ModuleType("utils.model_packages")
    packages.MODEL_PACKAGES = {
        "core": [],
        "advanced_ocr": [],
    }
    packages.PACKAGE_LABELS = {
        "core": ("Core", "Core modules"),
        "advanced_ocr": ("Advanced OCR", "Advanced OCR modules"),
    }
    sys.modules["utils.model_packages"] = packages
    return DummyMessageBox


def _load_dialog_module():
    _install_dialog_stubs()
    sys.modules.pop("ui.model_package_selector_dialog", None)
    return importlib.import_module("ui.model_package_selector_dialog")


def test_download_allows_advanced_only_with_confirmation():
    dlg_module = _load_dialog_module()
    dialog = dlg_module.ModelPackageSelectorDialog()
    dialog._checkboxes["core"].setChecked(False)
    dialog._checkboxes["advanced_ocr"].setChecked(True)

    dialog._on_download()

    assert dialog.get_selected_package_ids() == ["advanced_ocr"]


def test_download_defaults_to_core_when_none_selected():
    dlg_module = _load_dialog_module()
    dialog = dlg_module.ModelPackageSelectorDialog()
    dialog._checkboxes["core"].setChecked(False)
    dialog._checkboxes["advanced_ocr"].setChecked(False)

    dialog._on_download()

    assert dialog.get_selected_package_ids() == ["core"]
