"""
Compile Qt translation .ts files to .qm for the app to load.
Run from project root. Tries, in order:
  1. lrelease (Qt bin, e.g. from Qt SDK or Qt install)
  2. pylrelease6 (pip install pyqt6-tools)
  3. pyside6-lrelease (pip install pyside6)

If none is found, prints where to get lrelease and exits with 1.
"""
import os.path as osp
import subprocess
import sys
from glob import glob
import xml.etree.ElementTree as ET


def _lrelease_candidates(ts_path: str, qm_path: str):
    return [
        ['pyside6-lrelease', ts_path, '-qm', qm_path],
        ['lrelease', ts_path, '-qm', qm_path],
        ['lrelease-qt6', ts_path, '-qm', qm_path],
        [sys.executable, '-m', 'pylrelease6', ts_path, '-qm', qm_path],
        [sys.executable, '-m', 'PySide6.scripts.pyside_tool', 'lrelease', ts_path, '-qm', qm_path],
    ]


def _compile_one(root: str, ts_path: str, qm_path: str) -> bool:
    for cmd in _lrelease_candidates(ts_path, qm_path):
        try:
            r = subprocess.run(
                cmd,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f'[WARN] Compiler launch failed for {cmd}: {e}', file=sys.stderr)
            continue

        if r.returncode == 0 and osp.isfile(qm_path):
            print(f'Compiled: {qm_path}')
            return True

        stderr = (r.stderr or '').strip()
        if stderr:
            print(f'[DEBUG] failed cmd: {" ".join(cmd)}', file=sys.stderr)
            print(stderr, file=sys.stderr)
    return False



def _ts_translation_stats(ts_path: str) -> tuple[int, int, int]:
    """Return (total_messages, finished_messages, unfinished_messages)."""
    try:
        root = ET.parse(ts_path).getroot()
    except Exception:
        return (0, 0, 0)

    total = 0
    finished = 0
    unfinished = 0
    for msg in root.findall('.//message'):
        trans = msg.find('translation')
        if trans is None:
            continue
        total += 1
        is_unfinished = trans.get('type') == 'unfinished' or not (trans.text or '').strip()
        if is_unfinished:
            unfinished += 1
        else:
            finished += 1
    return total, finished, unfinished

def main():
    root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    trans_dir = osp.join(root, 'translate')
    if len(sys.argv) > 1:
        targets = []
        for arg in sys.argv[1:]:
            ts_path = arg if osp.isabs(arg) else osp.join(root, arg)
            if ts_path.endswith('.qm'):
                ts_path = ts_path[:-3] + '.ts'
            if not ts_path.endswith('.ts'):
                ts_path = ts_path + '.ts'
            if osp.isfile(ts_path):
                targets.append(ts_path)
            else:
                print(f'Not found: {ts_path}', file=sys.stderr)
    else:
        targets = sorted(glob(osp.join(trans_dir, '*.ts')))

    if not targets:
        print(f'No translation source files found in: {trans_dir}', file=sys.stderr)
        return 1

    ok = 0
    fail = 0
    stats = []
    for ts_path in targets:
        total, finished, unfinished = _ts_translation_stats(ts_path)
        stats.append((ts_path, total, finished, unfinished))
        qm_path = ts_path[:-3] + '.qm'
        if _compile_one(root, ts_path, qm_path):
            ok += 1
        else:
            fail += 1
            print(f'Failed to compile: {ts_path}', file=sys.stderr)

    for ts_path, total, finished, unfinished in stats:
        locale = osp.basename(ts_path).replace('.ts', '')
        if total:
            print(f'[i18n] {locale}: {finished}/{total} finished, {unfinished} unfinished')

    if fail == 0:
        print(f'All translations compiled successfully ({ok}/{ok}).')
        return 0

    print('Could not compile one or more translations. Compiler candidates tried:', file=sys.stderr)
    print('Install one of:', file=sys.stderr)
    print('  - Qt Linguist / Qt SDK (add bin/ to PATH), then run: lrelease translate/zh_CN.ts', file=sys.stderr)
    print('  - pip install pyqt6-tools  then: python -m pylrelease6 translate/zh_CN.ts -qm translate/zh_CN.qm', file=sys.stderr)
    print('  - pip install pyside6  then use pyside6-lrelease if available.', file=sys.stderr)
    print(f'Successful: {ok}, Failed: {fail}', file=sys.stderr)
    return 1

if __name__ == '__main__':
    sys.exit(main())
