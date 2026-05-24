import ast
from pathlib import Path


def test_welcome_widget_embeds_workflow_home_without_importing_qt():
    path = Path("ui/welcome_widget.py")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    assert "from .workflow_home import WorkflowHomeWidget" in text
    assert "self._workflow_home = WorkflowHomeWidget(parent=self)" in text
    assert "self._workflow_home.workflow_requested.connect(self._on_workflow_requested)" in text
    assert "open_assist_requested = Signal()" in text

    method_names = {
        node.name
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "WelcomeWidget"
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "_on_workflow_requested" in method_names


def test_welcome_workflow_dispatches_expected_signal_names():
    text = Path("ui/welcome_widget.py").read_text(encoding="utf-8")
    expected = [
        "open_live_requested.emit()",
        "open_images_requested.emit()",
        "open_downloader_requested.emit()",
        "open_batch_requested.emit()",
        "open_assist_requested.emit()",
        "open_models_requested.emit()",
        "open_diagnostics_requested.emit()",
    ]
    for snippet in expected:
        assert snippet in text
