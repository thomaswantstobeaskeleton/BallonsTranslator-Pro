import ast
from pathlib import Path


def test_experimental_shell_preview_dialog_exists_without_importing_qt():
    path = Path("ui/experimental_shell_preview_dialog.py")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "ExperimentalShellPreviewDialog" in class_names
    assert "open_experimental_shell_preview" in text


def test_experimental_shell_preview_wires_shell_signals():
    text = Path("ui/experimental_shell_preview_dialog.py").read_text(encoding="utf-8")
    expected = [
        "dashboard_action_requested",
        "workflow_requested",
        "mode_changed",
        "translation_assist_requested",
        "ocr_rerun_requested",
        "layout_review_requested",
        "typography_qa_requested",
        "job_cancel_requested",
        "job_pause_requested",
        "job_details_requested",
    ]
    for snippet in expected:
        assert snippet in text


def test_experimental_shell_preview_has_sample_jobs_button():
    text = Path("ui/experimental_shell_preview_dialog.py").read_text(encoding="utf-8")
    assert "Add sample jobs" in text
    assert "sample-ocr" in text
    assert "sample-export" in text
    assert "JobStatusSpec" in text
