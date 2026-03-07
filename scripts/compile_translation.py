"""
Compile Qt translation .ts files to .qm for the app to load.
Run from project root. Tries, in order:
  1. lrelease (Qt bin, e.g. from Qt SDK or Qt install)
  2. pylrelease6 (pip install pyqt6-tools)
  3. pyside6-lrelease (pip install pyside6)

If none is found, prints where to get lrelease and exits with 1.
"""
import os
import os.path as osp
import subprocess
import sys

def main():
    root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    trans_dir = osp.join(root, 'translate')
    ts_path = osp.join(trans_dir, 'zh_CN.ts')
    qm_path = osp.join(trans_dir, 'zh_CN.qm')

    if not osp.isfile(ts_path):
        print(f'Not found: {ts_path}', file=sys.stderr)
        return 1

    # 1) lrelease on PATH
    for cmd in ('lrelease', 'lrelease-qt6'):
        try:
            r = subprocess.run(
                [cmd, ts_path, '-qm', qm_path],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if r.returncode == 0:
                print(f'Compiled: {qm_path}')
                return 0
            if r.stderr and 'not found' not in r.stderr.lower():
                print(r.stderr, file=sys.stderr)
        except FileNotFoundError:
            pass

    # 2) pylrelease6
    try:
        r = subprocess.run(
            [sys.executable, '-m', 'pylrelease6', ts_path, '-qm', qm_path],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode == 0:
            print(f'Compiled: {qm_path}')
            return 0
    except Exception:
        pass

    # 3) pyside6-lrelease
    try:
        r = subprocess.run(
            [sys.executable, '-m', 'PySide6.scripts.pyside_tool', 'lrelease', ts_path, '-qm', qm_path],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode == 0:
            print(f'Compiled: {qm_path}')
            return 0
    except Exception:
        pass

    print('Could not find lrelease / pylrelease6 / pyside6-lrelease.', file=sys.stderr)
    print('Install one of:', file=sys.stderr)
    print('  - Qt Linguist / Qt SDK (add bin/ to PATH), then run: lrelease translate/zh_CN.ts', file=sys.stderr)
    print('  - pip install pyqt6-tools  then: python -m pylrelease6 translate/zh_CN.ts -qm translate/zh_CN.qm', file=sys.stderr)
    print('  - pip install pyside6  then use pyside6-lrelease if available.', file=sys.stderr)
    print('See docs/TRANSLATIONS.md for details.', file=sys.stderr)
    return 1

if __name__ == '__main__':
    sys.exit(main())
