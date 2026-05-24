import ast
from pathlib import Path


def _load_context_menu_items():
    src = Path("ui/context_menu_config_dialog.py").read_text(encoding="utf-8")
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CONTEXT_MENU_ITEMS":
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "CONTEXT_MENU_ITEMS":
            return ast.literal_eval(node.value)
    raise AssertionError("CONTEXT_MENU_ITEMS not found")


def test_context_menu_taxonomy_categories_present():
    CONTEXT_MENU_ITEMS = _load_context_menu_items()
    categories = [cat for cat, _items in CONTEXT_MENU_ITEMS]
    expected = {"Text box", "OCR", "Translation", "Style", "Layout", "QA", "Export"}
    assert expected.issubset(set(categories))


def test_context_menu_keys_unique_across_taxonomy():
    CONTEXT_MENU_ITEMS = _load_context_menu_items()
    keys = []
    for _cat, items in CONTEXT_MENU_ITEMS:
        for key, _label in items:
            keys.append(key)
    assert len(keys) == len(set(keys))
