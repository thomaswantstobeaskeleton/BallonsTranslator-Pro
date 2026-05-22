from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import QWidget

from .dashboard_action_dispatcher import dispatch_dashboard_action, dispatch_message_for_result
from .experimental_shell_menu import install_experimental_shell_preview_action


def install_experimental_shell_ui_hooks(mainwindow: QWidget) -> bool:
    """Install all safe experimental-shell hooks on a MainWindow-like object.

    This helper is designed to be called once after the title bar / Diagnostics
    menu exists. It keeps the actual MainWindow patch to a tiny guarded call:

        try:
            from ui.experimental_shell_bootstrap import install_experimental_shell_ui_hooks
            install_experimental_shell_ui_hooks(self)
        except Exception:
            pass

    The helper itself is safe to call more than once.
    """
    if mainwindow is None:
        return False
    action = install_experimental_shell_preview_action(mainwindow, mainwindow)
    return action is not None


def handle_experimental_dashboard_action(mainwindow: QWidget, mode: str, action: str) -> str:
    """Dispatch a dashboard action and return a user-visible status string."""
    result = dispatch_dashboard_action(mainwindow, mode, action)
    return dispatch_message_for_result(result)


def open_experimental_shell_preview_for(mainwindow: Optional[QWidget]):
    """Open the preview dialog and connect safe dashboard dispatch if possible."""
    from .experimental_shell_preview_dialog import ExperimentalShellPreviewDialog

    dialog = ExperimentalShellPreviewDialog(mainwindow)
    if mainwindow is not None:
        def _dispatch(mode: str, action: str):
            message = handle_experimental_dashboard_action(mainwindow, mode, action)
            status = getattr(mainwindow, "statusBar", lambda: None)()
            if status is not None and hasattr(status, "showMessage"):
                status.showMessage(message, 5000)
        dialog.dashboard_action_requested.connect(_dispatch)
        setattr(mainwindow, "_experimental_shell_preview_dialog", dialog)
    dialog.show()
    return dialog
