import ast
from pathlib import Path


def test_workflow_home_defines_expected_cards_without_importing_qt():
    path = Path("ui/workflow_home.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")

    expected_keys = [
        "editor",
        "live",
        "quick_image",
        "downloader",
        "batch",
        "assist",
        "models",
        "diagnostics",
    ]
    for key in expected_keys:
        assert f'key="{key}"' in text

    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "WorkflowCardSpec" in class_names
    assert "WorkflowCard" in class_names
    assert "WorkflowHomeWidget" in class_names


def test_workflow_home_uses_design_tokens():
    text = Path("ui/workflow_home.py").read_text(encoding="utf-8")
    assert "from .design_tokens import" in text
    assert "card_style" in text
    assert "badge_style" in text
    assert "WORKFLOW_ACCENTS" in text
