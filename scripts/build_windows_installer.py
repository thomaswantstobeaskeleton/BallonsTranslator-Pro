#!/usr/bin/env python3
"""Build Windows exe + installer (.exe) for BallonsTranslator-Pro.

Requires:
- Python + PyInstaller
- Inno Setup 6 (ISCC.exe available)
"""
import os
import os.path as osp
import subprocess
import sys


def _find_iscc() -> str:
    candidates = [
        os.environ.get("ISCC_EXE", ""),
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
    ]
    for p in candidates:
        if p and osp.isfile(p):
            return p
    raise FileNotFoundError("Inno Setup compiler (ISCC.exe) not found. Install Inno Setup 6 or set ISCC_EXE.")


def main() -> int:
    root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    os.chdir(root)
    py = sys.executable

    # 1) Build standalone exe with PyInstaller
    subprocess.check_call([py, "scripts/package_release.py"])

    # 2) Build installer wrapper using Inno Setup
    iss = osp.join("installer", "windows", "BallonsTranslatorPro.iss")
    iscc = _find_iscc()
    subprocess.check_call([iscc, iss])

    print("Installer build complete. Output in dist_installer/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
