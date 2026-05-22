import ast
from pathlib import Path


def test_experimental_app_shell_defines_expected_shell_pieces_without_importing_qt():
    path = Path("ui/experimental_app_shell.py")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "ShellPageSpec" in class_names
    assert "PlaceholderShellPage" in class_names
    assert "ExperimentalAppShell" in class_names

    for key in [
        "home",
        "editor",
        "live",
        "quick_image",
        "downloader",
        "batch",
        "assist",
        "models",
        "settings",
        "diagnostics",
    ]:
        assert f'ShellPageSpec("{key}"' in text


def test_experimental_app_shell_uses_expected_layout_components():
    text = Path("ui/experimental_app_shell.py").read_text(encoding="utf-8")
    expected = [
        "QSplitter",
        "QStackedWidget",
        "ModeRail",
        "WorkflowHomeWidget",
        "EditorInspector",
        "JobStatusDrawer",
        "save_splitter_state",
        "restore_splitter_state",
    ]
    for snippet in expected:
        assert snippet in text


def test_experimental_app_shell_exposes_routing_signals():
    text = Path("ui/experimental_app_shell.py").read_text(encoding="utf-8")
    expected = [
        "mode_changed = Signal(str)",
        "workflow_requested = Signal(str)",
        "translation_assist_requested = Signal()",
        "ocr_rerun_requested = Signal()",
        "layout_review_requested = Signal()",
        "typography_qa_requested = Signal()",
        "job_cancel_requested = Signal(str)",
        "job_pause_requested = Signal(str)",
        "job_details_requested = Signal(str)",
    ]
    for snippet in expected:
        assert snippet in text
