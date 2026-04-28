import ast
from pathlib import Path
import launch
import pytest


def _load_parse_model():
    src = Path('modules/textdetector/yolov5/yolo.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_model')
    text = ast.get_source_segment(src, fn)
    assert 'eval(m)' not in text
    assert 'eval(a)' not in text


def test_parse_model_has_no_eval_usage():
    _load_parse_model()


def test_launch_run_uses_argv_not_shell(monkeypatch):
    captured = {}
    class R:
        returncode = 0
        stdout = b'ok'
        stderr = b''
    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        return R()
    monkeypatch.setattr(launch.subprocess, 'run', fake_run)
    out = launch.run('python -c "print(1)"')
    assert isinstance(captured['cmd'], list)
    assert out == 'ok'
