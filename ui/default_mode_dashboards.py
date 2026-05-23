from __future__ import annotations

from typing import Iterable

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWidget

from .dashboard_action_dispatcher import dispatch_dashboard_action, dispatch_message_for_result
from .mode_dashboard import ModeDashboard, dashboard_for_mode

DEFAULT_DASHBOARD_MODES: tuple[tuple[str, str, str], ...] = (
    ("live", "Live Translation", "Realtime screen or Chrome manhua translation."),
    ("quick_image", "Image Quick Translation", "Drop images, translate quickly, and optionally promote to a project."),
    ("downloader", "Raw Downloader", "Search sources, choose chapters, and import raws."),
    ("batch", "Batch Queue", "Process many chapters, folders, or archives."),
    ("assist", "Translation Assist / QA", "Provider comparison, TM, glossary, concordance, SFX, and QA."),
    ("models", "Models & Providers", "Install models, test providers, and inspect runtime health."),
    ("diagnostics", "Diagnostics / Help", "Logs, doctors, reports, documentation, and support."),
)


def _log_dashboard_dispatch(mainwindow: QWidget, handled: bool, message: str):
    panel = getattr(mainwindow, "pipelineInsightsPanel", None)
    if panel is None:
        return
    try:
        if handled and hasattr(panel, "add_event"):
            panel.add_event("DASHBOARD", message)
        elif not handled and hasattr(panel, "add_warning"):
            panel.add_warning("DASHBOARD", message)
    except Exception:
        pass


def dispatch_default_dashboard_action(mainwindow: QWidget, mode: str, action: str):
    """Dispatch a ModeDashboard action through the existing action/router layer."""
    result = dispatch_dashboard_action(mainwindow, mode, action)
    _log_dashboard_dispatch(mainwindow, result.handled, dispatch_message_for_result(result))
    return result


def install_default_mode_dashboards(
    mainwindow: QWidget,
    *,
    modes: Iterable[tuple[str, str, str]] = DEFAULT_DASHBOARD_MODES,
    retry: bool = True,
) -> bool:
    """Install mode dashboard landing pages into MainWindow.centralStackWidget.

    This is intentionally additive: it appends dashboard pages after the existing
    welcome/editor/settings pages and records their indexes.  Existing handlers
    remain one click away through dashboard action buttons.
    """
    if mainwindow is None:
        return False
    if getattr(mainwindow, "_modern_dashboard_indexes", None) is not None:
        return True
    central_stack = getattr(mainwindow, "centralStackWidget", None)
    if central_stack is None or not hasattr(central_stack, "addWidget"):
        if retry:
            QTimer.singleShot(0, lambda: install_default_mode_dashboards(mainwindow, retry=False))
        return False

    indexes: dict[str, int] = {}
    for key, title, description in modes:
        spec = dashboard_for_mode(key, title, description)
        page = ModeDashboard(spec, central_stack)
        page.setObjectName(f"DefaultModeDashboard_{key}")
        page.action_requested.connect(lambda mode, action: dispatch_default_dashboard_action(mainwindow, mode, action))
        indexes[key] = central_stack.addWidget(page)
    mainwindow._modern_dashboard_indexes = indexes
    return True


def show_default_mode_dashboard(mainwindow: QWidget, mode: str) -> bool:
    """Show an installed dashboard page for a workflow mode."""
    indexes = getattr(mainwindow, "_modern_dashboard_indexes", None) or {}
    idx = indexes.get(str(mode or ""))
    if idx is None:
        return False
    central_stack = getattr(mainwindow, "centralStackWidget", None)
    if central_stack is None or not hasattr(central_stack, "setCurrentIndex"):
        return False
    try:
        central_stack.setCurrentIndex(idx)
        return True
    except Exception:
        return False
