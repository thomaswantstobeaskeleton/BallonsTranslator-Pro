from __future__ import annotations

from typing import Iterable, Sequence

from .job_status_drawer import JobStatusSpec
from .mode_dashboard import DashboardMetricSpec


def _first_attr(obj, names: Sequence[str], default=None):
    for name in names:
        try:
            value = getattr(obj, name, None)
        except Exception:
            value = None
        if value is not None:
            return value
    return default


def _safe_len(value) -> int | None:
    if value is None:
        return None
    try:
        return len(value)
    except Exception:
        return None


def _call_or_value(value):
    if callable(value):
        try:
            return value()
        except Exception:
            return None
    return value


def _project_page_count(mainwindow) -> int | None:
    candidates = (
        _first_attr(mainwindow, ("page_list", "pageList", "img_list", "imgList", "imgNameList", "pages")),
        _first_attr(_first_attr(mainwindow, ("proj", "project", "imgtrans_proj", "image_project")), ("pages", "page_list", "img_list")),
        _first_attr(_first_attr(mainwindow, ("imgtrans_proj", "project")), ("pages", "imgnames", "imgnames_list")),
    )
    for candidate in candidates:
        candidate = _call_or_value(candidate)
        length = _safe_len(candidate)
        if length is not None:
            return length
    return None


def _has_project(mainwindow) -> bool:
    checker = getattr(mainwindow, "_has_open_project", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            pass
    if _project_page_count(mainwindow):
        return True
    return bool(_first_attr(mainwindow, ("proj", "project", "imgtrans_proj", "proj_path", "project_path")))


def _warning_count(mainwindow) -> int:
    total = 0
    for owner_name in ("pipelineInsightsPanel", "maskDiagnosticsWidget", "typographyQAReport", "layoutReviewPanel"):
        owner = getattr(mainwindow, owner_name, None)
        if owner is None:
            continue
        for attr in ("warnings", "warning_items", "issues", "items"):
            value = _call_or_value(getattr(owner, attr, None))
            length = _safe_len(value)
            if length:
                total += length
                break
    drawer = getattr(mainwindow, "jobStatusDrawer", None)
    jobs = getattr(drawer, "_jobs", {}) if drawer is not None else {}
    try:
        total += sum(1 for job in jobs.values() if getattr(job, "warnings", None) or str(getattr(job, "status", "")).lower() in {"warning", "error", "failed"})
    except Exception:
        pass
    return total


def _job_summary(mainwindow) -> tuple[str, str, str]:
    drawer = getattr(mainwindow, "jobStatusDrawer", None)
    jobs: dict[str, JobStatusSpec] = getattr(drawer, "_jobs", {}) if drawer is not None else {}
    if not jobs:
        return "Idle", "idle", "No active jobs."
    running = [job for job in jobs.values() if str(getattr(job, "status", "")).lower() in {"running", "capturing", "ocr", "translating", "exporting"}]
    warning = [job for job in jobs.values() if getattr(job, "warnings", None) or str(getattr(job, "status", "")).lower() in {"warning", "error", "failed"}]
    if warning:
        return f"{len(warning)} need attention", "warning", f"{len(jobs)} total jobs."
    if running:
        return f"{len(running)} running", "running", f"{len(jobs)} total jobs."
    return f"{len(jobs)} tracked", "success", "All tracked jobs are complete or idle."


def _model_status(mainwindow) -> tuple[str, str, str]:
    manager = _first_attr(mainwindow, ("module_manager", "model_manager", "modelManager"))
    if manager is None:
        return "Unknown", "warning", "Model/provider status not available yet."
    missing = _first_attr(manager, ("missing_models", "missingModelList", "missing_modules"))
    missing = _call_or_value(missing)
    missing_count = _safe_len(missing)
    if missing_count:
        return f"{missing_count} missing", "warning", "Some optional models/providers may need setup."
    busy = _first_attr(manager, ("is_busy", "busy", "running"))
    busy = _call_or_value(busy)
    if busy:
        return "Busy", "running", "A model or provider task is currently running."
    return "Ready", "success", "No missing model/provider status reported."


def _source_status(mainwindow) -> tuple[str, str, str]:
    sources = _first_attr(mainwindow, ("manga_sources", "source_registry", "raw_sources", "mangaSourceList"))
    sources = _call_or_value(sources)
    count = _safe_len(sources)
    if count is not None:
        return str(count), "success" if count else "warning", "Registered raw/downloader sources."
    return "Configured", "idle", "Open the downloader to inspect source health."


def _translation_assist_status(mainwindow) -> tuple[str, str, str]:
    dock = _first_attr(mainwindow, ("translationAssistDock", "translation_assist_dock"))
    if dock is None:
        return "Available", "idle", "Open Assist to inspect candidates, TM, and glossary."
    visible = _call_or_value(getattr(dock, "isVisible", None))
    return ("Open" if visible else "Available", "success" if visible else "idle", "Translation Assist dock status.")


def metrics_for_mode(mainwindow, mode: str) -> list[DashboardMetricSpec]:
    mode = str(mode or "")
    project_open = _has_project(mainwindow)
    pages = _project_page_count(mainwindow)
    warnings = _warning_count(mainwindow)
    job_value, job_status, job_desc = _job_summary(mainwindow)
    model_value, model_status, model_desc = _model_status(mainwindow)

    if mode in {"editor", "quick_image"}:
        return [
            DashboardMetricSpec("pages", "Pages", str(pages) if pages is not None else ("Loaded" if project_open else "No project"), "success" if project_open else "idle", "Current project/page count."),
            DashboardMetricSpec("warnings", "Warnings", str(warnings), "warning" if warnings else "success", "Typography, mask, job, and export warnings."),
            DashboardMetricSpec("models", "Models", model_value, model_status, model_desc),
        ]
    if mode == "live":
        return [
            DashboardMetricSpec("capture", "Capture", job_value if job_value != "Idle" else "Idle", job_status, job_desc),
            DashboardMetricSpec("ocr", "OCR", model_value, model_status, model_desc),
            DashboardMetricSpec("latency", "Latency", "-- ms", "idle", "Realtime latency is shown after live mode starts."),
        ]
    if mode == "assist":
        assist_value, assist_status, assist_desc = _translation_assist_status(mainwindow)
        return [
            DashboardMetricSpec("candidates", "Candidates", assist_value, assist_status, assist_desc),
            DashboardMetricSpec("glossary", "Glossary", "Project", "idle", "Open Assist to load series glossary and TM."),
            DashboardMetricSpec("qa", "QA", str(warnings), "warning" if warnings else "success", "Project/block QA warning count."),
        ]
    if mode == "downloader":
        source_value, source_status, source_desc = _source_status(mainwindow)
        return [
            DashboardMetricSpec("sources", "Sources", source_value, source_status, source_desc),
            DashboardMetricSpec("queue", "Queue", job_value, job_status, job_desc),
            DashboardMetricSpec("rate", "Rate limits", "Safe", "success", "Downloader pacing status is conservative by default."),
        ]
    if mode in {"batch", "models", "diagnostics"}:
        return [
            DashboardMetricSpec("jobs", "Jobs", job_value, job_status, job_desc),
            DashboardMetricSpec("warnings", "Warnings", str(warnings), "warning" if warnings else "success", "Current warning count."),
            DashboardMetricSpec("models", "Models", model_value, model_status, model_desc),
        ]
    return []


def apply_dashboard_metrics(mainwindow, dashboard) -> bool:
    mode = getattr(getattr(dashboard, "spec", None), "key", "")
    metrics = metrics_for_mode(mainwindow, mode)
    if not metrics or not hasattr(dashboard, "update_metrics"):
        return False
    dashboard.update_metrics(metrics)
    return True


def refresh_default_dashboard_metrics(mainwindow) -> int:
    indexes = getattr(mainwindow, "_modern_dashboard_indexes", None) or {}
    central_stack = getattr(mainwindow, "centralStackWidget", None)
    if central_stack is None or not hasattr(central_stack, "widget"):
        return 0
    count = 0
    for _mode, index in indexes.items():
        try:
            dashboard = central_stack.widget(index)
        except Exception:
            dashboard = None
        if dashboard is not None and apply_dashboard_metrics(mainwindow, dashboard):
            count += 1
    return count
