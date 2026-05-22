import ast
from pathlib import Path


def test_model_manager_has_package_preset_combo_and_handler():
    src = Path('ui/model_manager_dialog.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    has_combo = False
    has_handler = False
    has_summary = False
    has_preset_only = False
    has_filter_method = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == 'packagePresetCombo':
            has_combo = True
        if isinstance(node, ast.Attribute) and node.attr == 'packagePresetSummary':
            has_summary = True
        if isinstance(node, ast.Attribute) and node.attr == 'showPresetOnlyCb':
            has_preset_only = True
        if isinstance(node, ast.FunctionDef) and node.name == '_on_package_preset_changed':
            has_handler = True
        if isinstance(node, ast.FunctionDef) and node.name == '_apply_download_filters':
            has_filter_method = True
    assert has_combo
    assert has_summary
    assert has_preset_only
    assert has_handler
    assert has_filter_method
