import ast
from pathlib import Path


def test_mangalens_detector_module_exists_and_registers_key():
    p = Path('modules/textdetector/detector_mangalens.py')
    assert p.exists()
    tree = ast.parse(p.read_text(encoding='utf-8'))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'register_textdetectors':
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'mangalens_bubble_segmentation':
                found = True
                break
    assert found
