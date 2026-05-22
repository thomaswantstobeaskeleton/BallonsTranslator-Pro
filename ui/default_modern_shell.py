from __future__ import annotations

from typing import Optional

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QHBoxLayout, QWidget

from .mode_rail import ModeRail


def _find_layout_containing_widget(root_layout, widget: QWidget) -> Optional[QHBoxLayout]:
    """Find the first nested layout that directly contains widget."""
    if root_layout is None or widget is None:
        return None
    try:
        if root_layout.indexOf(widget) >= 0:
            return root_layout
    except Exception:
        pass
    try:
        count = root_layout.count()
    except Exception:
        return None
    for idx in range(count):
        item = root_layout.itemAt(idx)
        if item is None:
            continue
        child_layout = item.layout()
        if child_layout is None:
            continue
        found = _find_layout_containing_widget(child_layout, widget)
        if found is not None:
            return found
    return None


def route_modern_mode_request(mainwindow: QWidget, mode: str):
    """Route ModeRail requests to existing MainWindow handlers.

    This keeps the new default rail useful immediately while preserving legacy
    left-bar/menu behavior during migration.
    """
    mode = str(mode or "").strip().lower()
    if not mainwindow:
        return
    if mode == "home" and hasattr(mainwindow, "_show_welcome_screen"):
        mainwindow._show_welcome_screen()
    elif mode == "editor":
        if hasattr(mainwindow, "_has_open_project") and mainwindow._has_open_project() and hasattr(mainwindow, "setupImgTransUI"):
            mainwindow.setupImgTransUI()
        elif hasattr(mainwindow, "_show_welcome_screen"):
            mainwindow._show_welcome_screen()
    elif mode == "live" and hasattr(mainwindow, "on_open_realtime_translator"):
        mainwindow.on_open_realtime_translator()
    elif mode == "quick_image" and hasattr(mainwindow, "leftBar") and hasattr(mainwindow.leftBar, "onOpenImages"):
        mainwindow.leftBar.onOpenImages()
    elif mode == "downloader" and hasattr(mainwindow, "on_open_manga_source"):
        mainwindow.on_open_manga_source()
    elif mode == "batch" and hasattr(mainwindow, "on_open_batch_queue"):
        mainwindow.on_open_batch_queue()
    elif mode == "assist" and hasattr(mainwindow, "on_open_translation_assist_dock"):
        mainwindow.on_open_translation_assist_dock()
    elif mode == "models" and hasattr(mainwindow, "on_open_manage_models"):
        mainwindow.on_open_manage_models()
    elif mode == "settings" and hasattr(mainwindow, "setupConfigUI"):
        mainwindow.setupConfigUI()
    elif mode == "diagnostics" and hasattr(mainwindow, "on_environment_doctor"):
        mainwindow.on_environment_doctor()


def set_modern_mode(mainwindow: QWidget, mode: str):
    rail = getattr(mainwindow, "modeRail", None)
    if rail is not None and hasattr(rail, "set_current_mode"):
        try:
            rail.set_current_mode(str(mode or ""))
        except Exception:
            pass


def install_default_modern_navigation(mainwindow: QWidget, *, retry: bool = True) -> bool:
    """Install ModeRail into the existing default MainWindow layout.

    This makes the rework default-facing without replacing the whole editor at
    once. It is safe to call more than once and safe to call before setupUi has
    fully finished; in that case it retries once on the next event-loop tick.
    """
    if mainwindow is None:
        return False
    if getattr(mainwindow, "modeRail", None) is not None:
        return True
    left_bar = getattr(mainwindow, "leftBar", None)
    root_layout = getattr(mainwindow, "mainvlayout", None)
    target_layout = _find_layout_containing_widget(root_layout, left_bar)
    if target_layout is None:
        if retry:
            QTimer.singleShot(0, lambda: install_default_modern_navigation(mainwindow, retry=False))
        return False
    rail = ModeRail(parent=mainwindow)
    rail.mode_requested.connect(lambda mode: route_modern_mode_request(mainwindow, mode))
    try:
        insert_at = max(0, target_layout.indexOf(left_bar))
    except Exception:
        insert_at = 0
    target_layout.insertWidget(insert_at, rail)
    mainwindow.modeRail = rail
    set_modern_mode(mainwindow, "home")
    return True


def install_default_welcome_signal_fallbacks(welcome_widget: QWidget, mainwindow: QWidget) -> bool:
    """Connect new welcome workflow signals to existing handlers when MainWindow has not yet done so."""
    if welcome_widget is None or mainwindow is None:
        return False
    if getattr(welcome_widget, "_modern_shell_fallbacks_installed", False):
        return True
    connected = False
    pairs = (
        ("open_assist_requested", "on_open_translation_assist_dock"),
    )
    for signal_name, handler_name in pairs:
        signal = getattr(welcome_widget, signal_name, None)
        handler = getattr(mainwindow, handler_name, None)
        if signal is not None and callable(handler):
            try:
                signal.connect(handler)
                connected = True
            except Exception:
                pass
    welcome_widget._modern_shell_fallbacks_installed = True
    return connected
