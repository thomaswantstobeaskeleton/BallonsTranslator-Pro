import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from qtpy.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget
except Exception as exc:  # pragma: no cover - optional Qt test environment
    pytest.skip(f"Qt is not available: {exc}", allow_module_level=True)

from ui.default_modern_shell import install_default_modern_navigation, route_modern_mode_request
from ui.mode_rail import ModeRail


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class DummyLeftBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.open_images_called = 0

    def onOpenImages(self):
        self.open_images_called += 1


class DummyMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.leftBar = DummyLeftBar(self)
        self.content = QWidget(self)
        self.mainvlayout = QVBoxLayout(self)
        self.main_hlayout = QHBoxLayout()
        self.main_hlayout.addWidget(self.leftBar)
        self.main_hlayout.addWidget(self.content)
        self.mainvlayout.addLayout(self.main_hlayout)
        self.project_open = False

    def _show_welcome_screen(self):
        self.calls.append("home")

    def _has_open_project(self):
        return self.project_open

    def setupImgTransUI(self):
        self.calls.append("editor")

    def on_open_realtime_translator(self):
        self.calls.append("live")

    def on_open_manga_source(self):
        self.calls.append("downloader")

    def on_open_batch_queue(self):
        self.calls.append("batch")

    def on_open_translation_assist_dock(self):
        self.calls.append("assist")

    def on_open_manage_models(self):
        self.calls.append("models")

    def setupConfigUI(self):
        self.calls.append("settings")

    def on_environment_doctor(self):
        self.calls.append("diagnostics")


def test_install_default_modern_navigation_adds_mode_rail_before_leftbar(qapp):
    win = DummyMainWindow()

    assert install_default_modern_navigation(win) is True

    assert isinstance(win.modeRail, ModeRail)
    assert win.main_hlayout.indexOf(win.modeRail) == 0
    assert win.main_hlayout.indexOf(win.leftBar) == 1
    assert win.modeRail.current_mode() == "home"


def test_install_default_modern_navigation_is_idempotent(qapp):
    win = DummyMainWindow()

    assert install_default_modern_navigation(win) is True
    first = win.modeRail
    assert install_default_modern_navigation(win) is True

    assert win.modeRail is first
    assert sum(1 for i in range(win.main_hlayout.count()) if win.main_hlayout.itemAt(i).widget() is first) == 1


def test_route_modern_mode_request_reuses_existing_handlers(qapp):
    win = DummyMainWindow()

    route_modern_mode_request(win, "home")
    route_modern_mode_request(win, "editor")
    win.project_open = True
    route_modern_mode_request(win, "editor")
    route_modern_mode_request(win, "quick_image")
    route_modern_mode_request(win, "assist")
    route_modern_mode_request(win, "diagnostics")

    assert win.calls == ["home", "home", "editor", "assist", "diagnostics"]
    assert win.leftBar.open_images_called == 1
