from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .job_status_drawer import JobStatusSpec


@dataclass(frozen=True)
class DashboardTaskJobSpec:
    mode: str
    action: str
    job_id: str
    title: str
    kind: str
    status: str = "running"
    progress: int = 5
    detail: str = ""
    can_cancel: bool = False


DASHBOARD_TASK_JOBS: Dict[Tuple[str, str], DashboardTaskJobSpec] = {
    ("editor", "run_pipeline"): DashboardTaskJobSpec(
        "editor",
        "run_pipeline",
        "pipeline-current",
        "Pipeline",
        "Pipeline",
        "running",
        1,
        "Pipeline launched from dashboard.",
        True,
    ),
    ("editor", "export_proof"): DashboardTaskJobSpec(
        "editor",
        "export_proof",
        "export-proof-pack",
        "Export proof pack",
        "Export",
        "running",
        10,
        "Proof-pack export launched from dashboard.",
        False,
    ),
    ("downloader", "primary"): DashboardTaskJobSpec(
        "downloader",
        "primary",
        "raw-downloader",
        "Raw downloader",
        "Download",
        "running",
        5,
        "Raw downloader opened. Select sources/chapters to start downloads.",
        False,
    ),
    ("downloader", "search"): DashboardTaskJobSpec(
        "downloader",
        "search",
        "raw-downloader",
        "Raw downloader",
        "Download",
        "running",
        5,
        "Raw source search opened from dashboard.",
        False,
    ),
    ("downloader", "queue"): DashboardTaskJobSpec(
        "downloader",
        "queue",
        "chapter-download-queue",
        "Chapter queue",
        "Queue",
        "running",
        5,
        "Chapter/download queue opened from dashboard.",
        False,
    ),
    ("downloader", "import"): DashboardTaskJobSpec(
        "downloader",
        "import",
        "raw-import",
        "Import raws",
        "Project",
        "running",
        10,
        "Import-to-project flow launched from dashboard.",
        False,
    ),
    ("batch", "primary"): DashboardTaskJobSpec(
        "batch",
        "primary",
        "batch-queue",
        "Batch queue",
        "Batch",
        "running",
        5,
        "Batch queue opened from dashboard.",
        True,
    ),
    ("models", "primary"): DashboardTaskJobSpec(
        "models",
        "primary",
        "model-manager",
        "Models & providers",
        "Models",
        "running",
        5,
        "Model/provider manager opened from dashboard.",
        False,
    ),
    ("assist", "primary"): DashboardTaskJobSpec(
        "assist",
        "primary",
        "translation-assist",
        "Translation Assist",
        "Assist",
        "running",
        5,
        "Translation Assist opened from dashboard.",
        False,
    ),
    ("assist", "compare"): DashboardTaskJobSpec(
        "assist",
        "compare",
        "translation-assist-compare",
        "Compare providers",
        "Assist",
        "running",
        10,
        "Provider comparison requested from dashboard.",
        False,
    ),
}


def dashboard_task_job_for(mode: str, action: str) -> Optional[DashboardTaskJobSpec]:
    return DASHBOARD_TASK_JOBS.get((str(mode or ""), str(action or "")))


def _drawer_for(mainwindow):
    return getattr(mainwindow, "jobStatusDrawer", None)


def upsert_task_job(mainwindow, spec: DashboardTaskJobSpec, *, handled: bool = True) -> bool:
    drawer = _drawer_for(mainwindow)
    if drawer is None or not hasattr(drawer, "upsert_job"):
        return False
    status = spec.status if handled else "warning"
    detail = spec.detail if handled else f"{spec.detail} Action did not complete; check the legacy UI."
    try:
        drawer.upsert_job(
            JobStatusSpec(
                job_id=spec.job_id,
                title=spec.title,
                kind=spec.kind,
                status=status,
                progress=max(0, min(100, int(spec.progress))),
                detail=detail,
                can_cancel=spec.can_cancel,
                warnings=[] if handled else ["Dashboard action was not handled by the existing UI."],
            )
        )
        return True
    except Exception:
        return False


def mirror_dashboard_task_job(mainwindow, mode: str, action: str, *, handled: bool = True) -> bool:
    spec = dashboard_task_job_for(mode, action)
    if spec is None:
        return False
    return upsert_task_job(mainwindow, spec, handled=handled)


def complete_task_job(mainwindow, job_id: str, *, detail: str = "Completed") -> bool:
    drawer = _drawer_for(mainwindow)
    if drawer is None or not hasattr(drawer, "_jobs") or not hasattr(drawer, "upsert_job"):
        return False
    current = getattr(drawer, "_jobs", {}).get(job_id)
    if current is None:
        return False
    try:
        drawer.upsert_job(
            JobStatusSpec(
                job_id=job_id,
                title=current.title,
                kind=current.kind,
                status="success",
                progress=100,
                detail=detail,
                can_cancel=False,
                warnings=list(current.warnings or []),
            )
        )
        return True
    except Exception:
        return False
