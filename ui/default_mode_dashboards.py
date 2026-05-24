from __future__ import annotations

from typing import Iterable

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWidget

from .dashboard_action_dispatcher import dispatch_dashboard_action, dispatch_message_for_result
from .dashboard_status_provider import apply_dashboard_metrics, refresh_default_dashboard_metrics
from .mode_dashboard import ModeDashboard, dashboard_for_mode
from .task_job_bridge import mirror_dashboard_task_job

DEFAULT_DASHBOARD_MODES: tuple[tuple[str, str, str], ...] = (
    ("live", "Live Translation", "Realtime screen or Chrome manhua translation."),
    ("quick_image", "Image Quick Translation", "Drop images, translate quickly, and optionally promote to a project."),
    ("downloader", "Raw Downloader", "Search sources, choose chapters, and import raws."),
    ("batch", "Batch Queue", "Process many chapters, folders, or archives."),
    ("assist", "Translation Assist / QA", "Provider comparison, TM, glossary, concordance, SFX, and QA."),
    ("models", "Models & Providers", "Install models, test providers, and inspect runtime health."),
    ("diagnostics", "Diagnostics / Help", "Logs, doctors, reports, documentation, and support."),
)
DASHBOARD_MODE_KEYS = frozenset(key for key, _title, _description in DEFAULT_DASHBOARD_MODES)


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


def _set_mode_rail(mainwindow: QWidget, mode: str):
    rail = getattr(mainwindow, "modeRail", None)
    if rail is not None and hasattr(rail, "set_current_mode"):
        try:
            rail.set_current_mode(str(mode or ""))
        except Exception:
            pass


def _set_workflow_hint(mainwindow: QWidget, mode: str):
    try:
        title_bar = getattr(mainwindow, "titleBar", None)
        if title_bar is not None and hasattr(title_bar, "set_workflow_hint"):
            title_bar.set_workflow_hint(str(mode or ""))
    except Exception:
        pass


def dispatch_default_dashboard_action(mainwindow: QWidget, mode: str, action: str):
    """Dispatch a ModeDashboard action and mirror task-like actions into the drawer."""
    result = dispatch_dashboard_action(mainwindow, mode, action)
    _log_dashboard_dispatch(mainwindow, result.handled, dispatch_message_for_result(result))
    mirror_dashboard_task_job(mainwindow, mode, action, handled=result.handled)
    refresh_default_dashboard_metrics(mainwindow)
    return result


def _legacy_route_for_mode(mainwindow: QWidget, mode: str) -> bool:
    """Fallback to existing handlers if dashboard pages are not installed."""
    if mode == "home" and hasattr(mainwindow, "_show_welcome_screen"):
        mainwindow._show_welcome_screen()
        refresh_default_dashboard_metrics(mainwindow)
        return True
    if mode == "editor":
        if hasattr(mainwindow, "_has_open_project") and mainwindow._has_open_project() and hasattr(mainwindow, "setupImgTransUI"):
            mainwindow.setupImgTransUI()
        elif hasattr(mainwindow, "_show_welcome_screen"):
            mainwindow._show_welcome_screen()
        refresh_default_dashboard_metrics(mainwindow)
        return True
    handler_map = {
        "live": "on_open_realtime_translator",
        "downloader": "on_open_manga_source",
        "batch": "on_open_batch_queue",
        "assist": "on_open_translation_assist_dock",
        "models": "on_open_manage_models",
        "settings": "setupConfigUI",
        "diagnostics": "on_environment_doctor",
    }
    if mode == "quick_image" and hasattr(mainwindow, "leftBar") and hasattr(mainwindow.leftBar, "onOpenImages"):
        mainwindow.leftBar.onOpenImages()
        _set_mode_rail(mainwindow, "quick_image")
        _set_workflow_hint(mainwindow, "quick_image")
        refresh_default_dashboard_metrics(mainwindow)
        return True
    handler_name = handler_map.get(mode)
    handler = getattr(mainwindow, handler_name, None) if handler_name else None
    if callable(handler):
        handler()
        refresh_default_dashboard_metrics(mainwindow)
        return True
    return False


def route_default_dashboard_mode(mainwindow: QWidget, mode: str) -> bool:
    """Route ModeRail requests to dashboard landing pages when available."""
    mode = str(mode or "").strip().lower()
    if not mainwindow:
        return False
    if mode == "home" or mode == "editor" or mode == "settings":
        return _legacy_route_for_mode(mainwindow, mode)
    if mode in DASHBOARD_MODE_KEYS and show_default_mode_dashboard(mainwindow, mode):
        _set_mode_rail(mainwindow, mode)
        _set_workflow_hint(mainwindow, mode)
        refresh_default_dashboard_metrics(mainwindow)
        return True
    return _legacy_route_for_mode(mainwindow, mode)


def _install_dashboard_rail_router(mainwindow: QWidget) -> bool:
    rail = getattr(mainwindow, "modeRail", None)
    if rail is None or not hasattr(rail, "mode_requested"):
        return False
    if getattr(mainwindow, "_modern_dashboard_rail_router_installed", False):
        return True
    try:
        rail.mode_requested.disconnect()
    except Exception:
        pass
    try:
        rail.mode_requested.connect(lambda mode: route_default_dashboard_mode(mainwindow, mode))
        mainwindow._modern_dashboard_rail_router_installed = True
        return True
    except Exception:
        return False


def install_default_mode_dashboards(
    mainwindow: QWidget,
    *,
    modes: Iterable[tuple[str, str, str]] = DEFAULT_DASHBOARD_MODES,
    retry: bool = True,
) -> bool:
    """Install mode dashboard landing pages into MainWindow.centralStackWidget.

    This is intentionally additive: it appends dashboard pages after the existing
    welcome/editor/settings pages and records their indexes. Existing handlers
    remain one click away through dashboard action buttons.
    """
    if mainwindow is None:
        return False
    central_stack = getattr(mainwindow, "centralStackWidget", None)
    if central_stack is None or not hasattr(central_stack, "addWidget"):
        if retry:
            QTimer.singleShot(0, lambda: install_default_mode_dashboards(mainwindow, retry=False))
        return False

    if getattr(mainwindow, "_modern_dashboard_indexes", None) is None:
        indexes: dict[str, int] = {}
        for key, title, description in modes:
            spec = dashboard_for_mode(key, title, description)
            page = ModeDashboard(spec, central_stack)
            page.setObjectName(f"DefaultModeDashboard_{key}")
            page.action_requested.connect(lambda mode, action: dispatch_default_dashboard_action(mainwindow, mode, action))
            apply_dashboard_metrics(mainwindow, page)
            indexes[key] = central_stack.addWidget(page)
        mainwindow._modern_dashboard_indexes = indexes
    else:
        refresh_default_dashboard_metrics(mainwindow)
    _install_dashboard_rail_router(mainwindow)
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
        dashboard = central_stack.widget(idx) if hasattr(central_stack, "widget") else None
        if dashboard is not None:
            apply_dashboard_metrics(mainwindow, dashboard)
        central_stack.setCurrentIndex(idx)
        return True
    except Exception:
        return False
