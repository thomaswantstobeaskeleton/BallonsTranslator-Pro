import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from qtpy.QtWidgets import QApplication, QStackedWidget, QWidget
except Exception as exc:  # pragma: no cover - optional Qt test environment
    pytest.skip(f"Qt is not available: {exc}", allow_module_level=True)

from ui.default_mode_dashboards import (
    DEFAULT_DASHBOARD_MODES,
    dispatch_default_dashboard_action,
    install_default_mode_dashboards,
    route_default_dashboard_mode,
    show_default_mode_dashboard,
)
from ui.mode_rail import ModeRail


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class FakeLeftBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.open_images_called = 0

    def onOpenImages(self):
        self.open_images_called += 1


class FakePipelinePanel:
    def __init__(self):
        self.events = []
        self.warnings = []

    def add_event(self, kind, message):
        self.events.append((kind, message))

    def add_warning(self, kind, message):
        self.warnings.append((kind, message))


class FakeTitleBar:
    def __init__(self):
        self.workflow_hints = []

    def set_workflow_hint(self, mode):
        self.workflow_hints.append(mode)


class DummyMainWindow(QWidget):
    def __init__(self, *, with_stack: bool = True, with_rail: bool = False):
        super().__init__()
        self.calls = []
        self.leftBar = FakeLeftBar(self)
        self.pipelineInsightsPanel = FakePipelinePanel()
        self.titleBar = FakeTitleBar()
        self.project_open = False
        if with_stack:
            self.centralStackWidget = QStackedWidget(self)
            # Existing default pages: welcome, editor, settings.
            self.centralStackWidget.addWidget(QWidget(self))
            self.centralStackWidget.addWidget(QWidget(self))
            self.centralStackWidget.addWidget(QWidget(self))
        if with_rail:
            self.modeRail = ModeRail(parent=self)

    def _show_welcome_screen(self):
        self.calls.append("home")

    def _has_open_project(self):
        return self.project_open

    def setupImgTransUI(self):
        self.calls.append("editor")

    def setupConfigUI(self):
        self.calls.append("settings")

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

    def on_environment_doctor(self):
        self.calls.append("diagnostics")


def test_install_default_mode_dashboards_appends_pages(qapp):
    win = DummyMainWindow(with_stack=True)
    before = win.centralStackWidget.count()

    assert install_default_mode_dashboards(win) is True

    indexes = win._modern_dashboard_indexes
    assert set(indexes) == {key for key, _title, _description in DEFAULT_DASHBOARD_MODES}
    assert win.centralStackWidget.count() == before + len(DEFAULT_DASHBOARD_MODES)


def test_install_default_mode_dashboards_is_idempotent(qapp):
    win = DummyMainWindow(with_stack=True)

    assert install_default_mode_dashboards(win) is True
    first_indexes = dict(win._modern_dashboard_indexes)
    first_count = win.centralStackWidget.count()
    assert install_default_mode_dashboards(win) is True

    assert win._modern_dashboard_indexes == first_indexes
    assert win.centralStackWidget.count() == first_count


def test_show_default_mode_dashboard_switches_central_stack(qapp):
    win = DummyMainWindow(with_stack=True)
    install_default_mode_dashboards(win)

    assert show_default_mode_dashboard(win, "live") is True
    assert win.centralStackWidget.currentIndex() == win._modern_dashboard_indexes["live"]
    assert show_default_mode_dashboard(win, "missing") is False


def test_route_default_dashboard_mode_prefers_dashboard_page(qapp):
    win = DummyMainWindow(with_stack=True)
    install_default_mode_dashboards(win)

    assert route_default_dashboard_mode(win, "downloader") is True

    assert win.calls == []
    assert win.centralStackWidget.currentIndex() == win._modern_dashboard_indexes["downloader"]
    assert win.titleBar.workflow_hints[-1] == "downloader"


def test_route_default_dashboard_mode_falls_back_without_dashboards(qapp):
    win = DummyMainWindow(with_stack=False)

    assert route_default_dashboard_mode(win, "live") is True

    assert win.calls == ["live"]


def test_installed_mode_rail_routes_to_dashboard_page(qapp):
    win = DummyMainWindow(with_stack=True, with_rail=True)
    install_default_mode_dashboards(win)

    win.modeRail.mode_requested.emit("assist")

    assert win.centralStackWidget.currentIndex() == win._modern_dashboard_indexes["assist"]
    assert win.modeRail.current_mode() == "assist"
    assert win.calls == []


def test_dashboard_action_dispatch_uses_existing_child_handler_and_logs(qapp):
    win = DummyMainWindow(with_stack=True)

    result = dispatch_default_dashboard_action(win, "quick_image", "primary")

    assert result.handled is True
    assert result.invoked == "handler:leftBar.onOpenImages"
    assert win.leftBar.open_images_called == 1
    assert win.pipelineInsightsPanel.events[-1][0] == "DASHBOARD"
