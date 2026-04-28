import os

import pytest

from utils import windows_context_menu


def test_get_command_snapshot_uses_pythonw_when_available(monkeypatch):
    repo_root = "/repo"
    python_exe = "/opt/python/python.exe"
    pythonw_exe = "/opt/python/pythonw.exe"

    monkeypatch.setattr(windows_context_menu.sys, "executable", python_exe)

    launch_py = os.path.join(repo_root, "launch.py")

    def fake_isfile(path):
        return path in {launch_py, pythonw_exe}

    monkeypatch.setattr(windows_context_menu.os.path, "isfile", fake_isfile)

    cmd = windows_context_menu.get_command(repo_root)
    assert cmd == f'"{pythonw_exe}" "{launch_py}" "--proj-dir" "%1"'


def test_get_command_snapshot_falls_back_to_python_exe(monkeypatch):
    repo_root = "/repo"
    python_exe = "/opt/venv/bin/python"

    monkeypatch.setattr(windows_context_menu.sys, "executable", python_exe)

    launch_py = os.path.join(repo_root, "launch.py")
    monkeypatch.setattr(windows_context_menu.os.path, "isfile", lambda path: path == launch_py)

    cmd = windows_context_menu.get_command(repo_root)
    assert cmd == f'"{python_exe}" "{launch_py}" "--proj-dir" "%1"'


def test_get_command_raises_when_launch_py_missing(monkeypatch):
    monkeypatch.setattr(windows_context_menu.os.path, "isfile", lambda _path: False)

    with pytest.raises(FileNotFoundError):
        windows_context_menu.get_command("/missing")
