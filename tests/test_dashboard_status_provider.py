from ui.dashboard_status_provider import metrics_for_mode, refresh_default_dashboard_metrics
from ui.job_status_drawer import JobStatusSpec
from ui.mode_dashboard import ModeDashboard, dashboard_for_mode


class FakeDrawer:
    def __init__(self, jobs=None):
        self._jobs = jobs or {}


class FakeManager:
    def __init__(self, missing=None, busy=False):
        self.missing_models = missing or []
        self.is_busy = busy


class FakeCentralStack:
    def __init__(self):
        self._widgets = []

    def addWidget(self, widget):
        self._widgets.append(widget)
        return len(self._widgets) - 1

    def widget(self, index):
        return self._widgets[index]


class FakeMainWindow:
    def __init__(self):
        self.page_list = []
        self.module_manager = FakeManager()
        self.jobStatusDrawer = FakeDrawer()

    def _has_open_project(self):
        return bool(self.page_list)


def metric_by_key(metrics, key):
    for metric in metrics:
        if metric.key == key:
            return metric
    raise AssertionError(f"missing metric {key}")


def test_editor_metrics_reflect_project_pages_and_warnings():
    win = FakeMainWindow()
    win.page_list = ["001.png", "002.png"]
    win.jobStatusDrawer = FakeDrawer({
        "pipeline": JobStatusSpec(job_id="pipeline", title="Pipeline", status="warning", warnings=["overflow"])
    })

    metrics = metrics_for_mode(win, "editor")

    assert metric_by_key(metrics, "pages").value == "2"
    assert metric_by_key(metrics, "pages").status == "success"
    assert metric_by_key(metrics, "warnings").value == "1"
    assert metric_by_key(metrics, "warnings").status == "warning"


def test_model_metric_reports_missing_models():
    win = FakeMainWindow()
    win.module_manager = FakeManager(missing=["ocr", "detector"])

    metrics = metrics_for_mode(win, "models")

    assert metric_by_key(metrics, "models").value == "2 missing"
    assert metric_by_key(metrics, "models").status == "warning"


def test_job_summary_reports_running_jobs():
    win = FakeMainWindow()
    win.jobStatusDrawer = FakeDrawer({
        "pipeline": JobStatusSpec(job_id="pipeline", title="Pipeline", status="running", progress=50)
    })

    metrics = metrics_for_mode(win, "batch")

    assert metric_by_key(metrics, "jobs").value == "1 running"
    assert metric_by_key(metrics, "jobs").status == "running"


def test_refresh_default_dashboard_metrics_updates_installed_dashboards():
    win = FakeMainWindow()
    win.page_list = ["001.png"]
    win.centralStackWidget = FakeCentralStack()
    dashboard = ModeDashboard(dashboard_for_mode("quick_image", "Quick", "Quick images"))
    idx = win.centralStackWidget.addWidget(dashboard)
    win._modern_dashboard_indexes = {"quick_image": idx}

    assert refresh_default_dashboard_metrics(win) == 1

    snapshot = dashboard.metric_snapshot()
    assert snapshot["pages"].value == "1"
    assert snapshot["pages"].status == "success"
