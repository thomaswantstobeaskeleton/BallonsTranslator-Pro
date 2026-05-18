#!/usr/bin/env python3
"""One-command packaging helper for Windows/macOS/Linux.

Builds a desktop executable via PyInstaller using `launch.spec`.
"""
import os
import os.path as osp
import subprocess
import sys


MIN_PYINSTALLER = "6.11"
PYINSTALLER_SPEC = f"pyinstaller>={MIN_PYINSTALLER},<7"


def _check_supported_python() -> None:
    """Fail early for CPython releases known to crash PyInstaller analysis."""
    if (3, 10, 0) <= sys.version_info[:3] < (3, 10, 2):
        version = ".".join(str(part) for part in sys.version_info[:3])
        raise SystemExit(
            "Python 3.10 builds require Python 3.10.2 or newer. "
            f"You are running Python {version} from {sys.executable}.\n"
            "Python 3.10.0/3.10.1 can crash PyInstaller while disassembling "
            "modules with `IndexError: tuple index out of range`. Install a "
            "current Python 3.10+ release, then rerun build_windows_installer.bat."
        )


def _install_build_dependencies(python: str) -> None:
    subprocess.check_call([python, "-m", "pip", "install", "-r", "requirements.txt"])
    subprocess.check_call([python, "-m", "pip", "install", "--upgrade", PYINSTALLER_SPEC])


def main() -> int:
    root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    os.chdir(root)
    _check_supported_python()
    python = sys.executable
    _install_build_dependencies(python)
    subprocess.check_call([python, "-m", "PyInstaller", "--clean", "--noconfirm", "launch.spec"])
    print("Build complete. Check dist/ for executable bundle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
