#!/usr/bin/env python3
"""One-command packaging helper for Windows/macOS/Linux.

Builds a desktop executable via PyInstaller using `launch.spec`.
"""
import os
import os.path as osp
import subprocess
import sys


def main() -> int:
    root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    os.chdir(root)
    python = sys.executable
    subprocess.check_call([python, "-m", "pip", "install", "-r", "requirements.txt"])
    subprocess.check_call([python, "-m", "pip", "install", "pyinstaller"])
    subprocess.check_call([python, "-m", "PyInstaller", "--noconfirm", "launch.spec"])
    print("Build complete. Check dist/ for executable bundle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
