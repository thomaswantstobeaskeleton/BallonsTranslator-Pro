from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import QAction, QMenu, QWidget


EXPERIMENTAL_SHELL_PREVIEW_ACTION_ID = "diagnostics.experimental_shell_preview"
EXPERIMENTAL_SHELL_PREVIEW_TEXT = "Experimental advanced UI shell preview..."


def _find_diagnostics_menu(owner: object) -> Optional[QMenu]:
    """Best-effort diagnostics menu lookup for TitleBar/MainWindow-like owners."""
    candidates = [owner]
    for attr in ("titleBar", "titlebar", "title_bar"):
        holder = getattr(owner, attr, None)
        if holder is not None:
            candidates.append(holder)
    for obj in candidates:
        btn = getattr(obj, "diagnosticsToolBtn", None)
        if btn is not None and hasattr(btn, "menu"):
            menu = btn.menu()
            if isinstance(menu, QMenu):
                return menu
        menu = getattr(obj, "diagnosticsMenu", None)
        if isinstance(menu, QMenu):
            return menu
    return None


def _register_with_action_registry(owner: object, action: QAction):
    """Best-effort action registry registration without depending on registry internals."""
    candidates = [owner]
    for attr in ("titleBar", "titlebar", "title_bar"):
        holder = getattr(owner, attr, None)
        if holder is not None:
            candidates.append(holder)
    for obj in candidates:
        registry = getattr(obj, "_action_registry", None) or getattr(obj, "action_registry", None)
        if registry is None:
            continue
        # Prefer a specific register method when present. If not, the existing
        # menu-tree scan will discover the QAction once it is added to the menu.
        for method_name in ("register_action", "add_action"):
            method = getattr(registry, method_name, None)
            if callable(method):
                try:
                    method(
                        action_id=EXPERIMENTAL_SHELL_PREVIEW_ACTION_ID,
                        action=action,
                        label=EXPERIMENTAL_SHELL_PREVIEW_TEXT,
                        category="Diagnostics",
                        workflow_mode="diagnostics",
                    )
                    return True
                except TypeError:
                    try:
                        method(EXPERIMENTAL_SHELL_PREVIEW_ACTION_ID, action)
                        return True
                    except Exception:
                        pass
                except Exception:
                    pass
    return False


def install_experimental_shell_preview_action(owner: QWidget, parent: Optional[QWidget] = None) -> Optional[QAction]:
    """Install a Diagnostics menu action that opens the experimental shell preview.

    This installer is intentionally separate from `mainwindowbars.py` so the
    preview command can be introduced without replacing a large title-bar file.
    It is safe to call more than once: if an action with the same text already
    exists in the diagnostics menu, that action is returned.
    """
    menu = _find_diagnostics_menu(owner)
    if menu is None:
        return None

    for existing in menu.actions() or []:
        if (existing.text() or "").replace("&", "").strip() == EXPERIMENTAL_SHELL_PREVIEW_TEXT:
            return existing

    action_parent = parent or owner
    action = QAction(EXPERIMENTAL_SHELL_PREVIEW_TEXT, action_parent)
    action.setObjectName("ExperimentalShellPreviewAction")
    action.setToolTip("Open the modern advanced UI shell preview without replacing the current editor.")

    def _open_preview():
        try:
            from .experimental_shell_preview_dialog import ExperimentalShellPreviewDialog
            dialog = ExperimentalShellPreviewDialog(owner)
            setattr(owner, "_experimental_shell_preview_dialog", dialog)
            dialog.show()
        except Exception as exc:
            try:
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.warning(owner, "Experimental shell preview", f"Failed to open preview: {exc}")
            except Exception:
                raise

    action.triggered.connect(_open_preview)
    menu.addSeparator()
    menu.addAction(action)
    setattr(owner, "experimental_shell_preview_trigger", action.triggered)
    _register_with_action_registry(owner, action)
    return action
