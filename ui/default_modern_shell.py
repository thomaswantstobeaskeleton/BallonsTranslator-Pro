from __future__ import annotations

from types import MethodType
from typing import Optional

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QHBoxLayout, QWidget

from .job_status_drawer import JobStatusDrawer, JobStatusSpec
from .mode_rail import ModeRail

PIPELINE_JOB_ID = "pipeline-current"


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
        set_modern_mode(mainwindow, "quick_image")
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


def _install_mode_sync_wrappers(mainwindow: QWidget):
    """Keep ModeRail checked state in sync when legacy menus/buttons navigate.

    During migration the old left bar, top menus, welcome cards, and shortcuts can
    all still change the current workspace.  Wrapping the existing handlers keeps
    the new rail accurate without rewriting every call site in MainWindow.
    """
    if mainwindow is None or getattr(mainwindow, "_modern_mode_sync_wrappers_installed", False):
        return

    def wrap(handler_name: str, mode: str, *, editor_requires_project: bool = False):
        original = getattr(mainwindow, handler_name, None)
        if not callable(original):
            return

        def wrapped(self, *args, __original=original, __mode=mode, __editor_requires_project=editor_requires_project, **kwargs):
            result = __original(*args, **kwargs)
            resolved_mode = __mode
            if __editor_requires_project:
                try:
                    resolved_mode = "editor" if self._has_open_project() else "home"
                except Exception:
                    resolved_mode = "home"
            set_modern_mode(self, resolved_mode)
            return result

        try:
            setattr(mainwindow, handler_name, MethodType(wrapped, mainwindow))
        except Exception:
            pass

    wrap("_show_welcome_screen", "home")
    wrap("_show_main_content", "editor", editor_requires_project=True)
    wrap("setupImgTransUI", "editor", editor_requires_project=True)
    wrap("setupConfigUI", "settings")
    wrap("on_open_realtime_translator", "live")
    wrap("on_open_manga_source", "downloader")
    wrap("on_open_batch_queue", "batch")
    wrap("on_open_translation_assist_dock", "assist")
    wrap("on_open_manage_models", "models")
    wrap("on_environment_doctor", "diagnostics")
    mainwindow._modern_mode_sync_wrappers_installed = True


def _record_job_drawer_event(mainwindow: QWidget, event_type: str, job_id: str):
    """Best-effort bridge from the new drawer buttons to existing status surfaces."""
    label = f"{event_type}: {job_id}"
    panel = getattr(mainwindow, "pipelineInsightsPanel", None)
    if panel is not None:
        try:
            if event_type == "cancel" and hasattr(panel, "add_warning"):
                panel.add_warning("JOB", label)
            elif hasattr(panel, "add_event"):
                panel.add_event("JOB", label)
        except Exception:
            pass
    if event_type == "cancel":
        try:
            module_manager = getattr(mainwindow, "module_manager", None)
            if module_manager is not None and hasattr(module_manager, "stopImgtransPipeline"):
                module_manager.stopImgtransPipeline()
        except Exception:
            pass


def upsert_default_job(mainwindow: QWidget, job: JobStatusSpec) -> bool:
    """Add/update a job in the default drawer when installed."""
    drawer = getattr(mainwindow, "jobStatusDrawer", None)
    if drawer is None:
        return False
    try:
        drawer.upsert_job(job)
        return True
    except Exception:
        return False


def _install_job_event_wrappers(mainwindow: QWidget):
    """Mirror legacy pipeline events into JobStatusDrawer without changing pipeline logic."""
    if mainwindow is None or getattr(mainwindow, "_modern_job_event_wrappers_installed", False):
        return

    stage_original = getattr(mainwindow, "on_pipeline_stage_event", None)
    if callable(stage_original):
        def stage_wrapped(self, stage_name: str, progress: int, page_name: str, __original=stage_original):
            result = __original(stage_name, progress, page_name)
            try:
                stage = str(stage_name or "Pipeline")
                page = str(page_name or "")
                detail = f"{stage} · {page}" if page else stage
                upsert_default_job(
                    self,
                    JobStatusSpec(
                        job_id=PIPELINE_JOB_ID,
                        title="Pipeline",
                        kind="pipeline",
                        status="running",
                        progress=max(0, min(100, int(progress or 0))),
                        detail=detail,
                        can_cancel=True,
                        can_pause=False,
                    ),
                )
            except Exception:
                pass
            return result

        try:
            setattr(mainwindow, "on_pipeline_stage_event", MethodType(stage_wrapped, mainwindow))
        except Exception:
            pass

    finish_original = getattr(mainwindow, "on_imgtrans_pipeline_finished", None)
    if callable(finish_original):
        def finish_wrapped(self, *args, __original=finish_original, **kwargs):
            result = __original(*args, **kwargs)
            try:
                upsert_default_job(
                    self,
                    JobStatusSpec(
                        job_id=PIPELINE_JOB_ID,
                        title="Pipeline",
                        kind="pipeline",
                        status="succeeded",
                        progress=100,
                        detail="Pipeline finished",
                        can_cancel=False,
                    ),
                )
            except Exception:
                pass
            return result

        try:
            setattr(mainwindow, "on_imgtrans_pipeline_finished", MethodType(finish_wrapped, mainwindow))
        except Exception:
            pass

    mainwindow._modern_job_event_wrappers_installed = True


def install_default_job_drawer(mainwindow: QWidget, *, retry: bool = True) -> bool:
    """Install the collapsed bottom Jobs & Status drawer into the default layout.

    The drawer starts collapsed and does not replace existing progress dialogs.
    It gives the modern shell a persistent status surface while later PRs wire
    concrete OCR/translation/inpaint/export/download jobs into it.
    """
    if mainwindow is None:
        return False
    if getattr(mainwindow, "jobStatusDrawer", None) is not None:
        _install_job_event_wrappers(mainwindow)
        return True
    root_layout = getattr(mainwindow, "mainvlayout", None)
    if root_layout is None:
        if retry:
            QTimer.singleShot(0, lambda: install_default_job_drawer(mainwindow, retry=False))
        return False
    drawer = JobStatusDrawer(parent=mainwindow)
    drawer.setObjectName("DefaultJobStatusDrawer")
    drawer.set_expanded(False)
    drawer.cancel_requested.connect(lambda job_id: _record_job_drawer_event(mainwindow, "cancel", job_id))
    drawer.pause_requested.connect(lambda job_id: _record_job_drawer_event(mainwindow, "pause", job_id))
    drawer.details_requested.connect(lambda job_id: _record_job_drawer_event(mainwindow, "details", job_id))
    insert_at = None
    bottom_bar = getattr(mainwindow, "bottomBar", None)
    if bottom_bar is not None:
        try:
            idx = root_layout.indexOf(bottom_bar)
            if idx >= 0:
                insert_at = idx
        except Exception:
            insert_at = None
    if insert_at is None:
        try:
            insert_at = max(0, root_layout.count())
        except Exception:
            insert_at = 0
    root_layout.insertWidget(insert_at, drawer)
    mainwindow.jobStatusDrawer = drawer
    _install_job_event_wrappers(mainwindow)
    return True


def install_default_modern_navigation(mainwindow: QWidget, *, retry: bool = True) -> bool:
    """Install ModeRail into the existing default MainWindow layout.

    This makes the rework default-facing without replacing the whole editor at
    once. It is safe to call more than once and safe to call before setupUi has
    fully finished; in that case it retries once on the next event-loop tick.
    """
    if mainwindow is None:
        return False
    if getattr(mainwindow, "modeRail", None) is not None:
        _install_mode_sync_wrappers(mainwindow)
        install_default_job_drawer(mainwindow, retry=retry)
        return True
    left_bar = getattr(mainwindow, "leftBar", None)
    root_layout = getattr(mainwindow, "mainvlayout", None)
    target_layout = _find_layout_containing_widget(root_layout, left_bar)
    if target_layout is None:
        if retry:
            QTimer.singleShot(0, lambda: install_default_modern_navigation(mainwindow, retry=False))
            QTimer.singleShot(0, lambda: install_default_job_drawer(mainwindow, retry=False))
        return False
    rail = ModeRail(parent=mainwindow)
    rail.mode_requested.connect(lambda mode: route_modern_mode_request(mainwindow, mode))
    try:
        insert_at = max(0, target_layout.indexOf(left_bar))
    except Exception:
        insert_at = 0
    target_layout.insertWidget(insert_at, rail)
    mainwindow.modeRail = rail
    _install_mode_sync_wrappers(mainwindow)
    install_default_job_drawer(mainwindow, retry=retry)
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
