from ui.job_status_drawer import JobStatusSpec
from ui.task_job_bridge import (
    complete_task_job,
    dashboard_task_job_for,
    mirror_dashboard_task_job,
    upsert_task_job,
)


class FakeDrawer:
    def __init__(self):
        self._jobs = {}

    def upsert_job(self, job: JobStatusSpec):
        self._jobs[job.job_id] = job


class FakeMainWindow:
    def __init__(self):
        self.jobStatusDrawer = FakeDrawer()


def test_dashboard_task_job_for_known_export_action():
    spec = dashboard_task_job_for("editor", "export_proof")

    assert spec is not None
    assert spec.job_id == "export-proof-pack"
    assert spec.kind == "Export"


def test_mirror_dashboard_task_job_upserts_handled_job():
    win = FakeMainWindow()

    assert mirror_dashboard_task_job(win, "models", "primary", handled=True) is True

    job = win.jobStatusDrawer._jobs["model-manager"]
    assert job.title == "Models & providers"
    assert job.status == "running"
    assert job.progress == 5
    assert job.warnings == []


def test_mirror_dashboard_task_job_marks_unhandled_as_warning():
    win = FakeMainWindow()

    assert mirror_dashboard_task_job(win, "assist", "compare", handled=False) is True

    job = win.jobStatusDrawer._jobs["translation-assist-compare"]
    assert job.status == "warning"
    assert job.warnings == ["Dashboard action was not handled by the existing UI."]


def test_mirror_dashboard_task_job_ignores_unknown_action():
    win = FakeMainWindow()

    assert mirror_dashboard_task_job(win, "missing", "action", handled=True) is False
    assert win.jobStatusDrawer._jobs == {}


def test_complete_task_job_marks_existing_job_success():
    win = FakeMainWindow()
    spec = dashboard_task_job_for("batch", "primary")
    assert upsert_task_job(win, spec, handled=True) is True

    assert complete_task_job(win, "batch-queue", detail="Batch queue finished") is True

    job = win.jobStatusDrawer._jobs["batch-queue"]
    assert job.status == "success"
    assert job.progress == 100
    assert job.detail == "Batch queue finished"
    assert job.can_cancel is False
