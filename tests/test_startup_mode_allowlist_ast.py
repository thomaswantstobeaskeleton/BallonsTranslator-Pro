import ast
from pathlib import Path


def test_startup_mode_allowlist_includes_all_routed_modes():
    src = Path('utils/config.py').read_text(encoding='utf-8')
    mod = ast.parse(src)
    target = None
    for node in ast.walk(mod):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Call):
            fn = node.left.func
            if isinstance(fn, ast.Attribute) and fn.attr == 'get':
                if node.left.args and isinstance(node.left.args[0], ast.Constant) and node.left.args[0].value == 'startup_mode':
                    if node.comparators and isinstance(node.comparators[0], ast.Set):
                        target = {elt.value for elt in node.comparators[0].elts if isinstance(elt, ast.Constant)}
                        break
    assert target is not None
    expected = {'home', 'editor', 'last_used', 'settings', 'live', 'downloader', 'batch', 'models', 'diagnostics'}
    assert expected.issubset(target)
