import ast
from pathlib import Path


def test_job_status_drawer_defines_shell_classes_without_importing_qt():
    path = Path("ui/job_status_drawer.py")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "JobStatusSpec" in class_names
    assert "JobStatusRow" in class_names
    assert "JobStatusDrawer" in class_names


def test_job_status_drawer_has_required_signals_and_methods():
    text = Path("ui/job_status_drawer.py").read_text(encoding="utf-8")
    required = [
        "cancel_requested = Signal(str)",
        "pause_requested = Signal(str)",
        "details_requested = Signal(str)",
        "clear_completed_requested = Signal()",
        "def refresh_jobs",
        "def upsert_job",
        "def remove_job",
        "def set_expanded",
    ]
    for snippet in required:
        assert snippet in text


def test_job_status_drawer_mentions_core_job_types():
    text = Path("ui/job_status_drawer.py").read_text(encoding="utf-8")
    expected = ["OCR", "translation", "inpaint", "render", "export", "raw download", "live translation", "model download", "batch queue", "Translation"]
    for word in expected:
        assert word in text
