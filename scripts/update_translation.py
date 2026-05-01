"""
Update Qt translation .ts files from UI sources.

Run from project root. Requires: pylupdate6 (PyQt6) or pylupdate5 (PyQt5).
By default, updates every existing `translate/*.ts` locale file so non-English
locales stay in sync when new `self.tr(...)` strings are added in UI code.

Usage:
  python scripts/update_translation.py                # update all translate/*.ts
  python scripts/update_translation.py --locale zh_CN # update only one locale

Compile to .qm with: lrelease translate/<locale>.ts
"""
import os
import os.path as osp
import argparse
from glob import glob


def _run_update(program_dir: str, ui_list: str, ts_path: str) -> int:
    cmd = f'pylupdate6 --no-obsolete --verbose {ui_list} -ts "{ts_path}"'
    try:
        import subprocess
        result = subprocess.run(cmd, shell=True, cwd=program_dir)
        if result.returncode == 0:
            return 0
    except Exception:
        pass

    cmd5 = f'pylupdate5 --no-obsolete --verbose {ui_list} -ts "{ts_path}"'
    return os.system(cmd5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update Qt translation ts files from ui/**/*.py')
    parser.add_argument('--locale', default='', help='Locale to update (e.g. zh_CN). Default: all translate/*.ts')
    args = parser.parse_args()

    program_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    translate_dir = osp.join(program_dir, 'translate')
    ui_files = (
        glob(osp.join(program_dir, 'ui', '*.py')) +
        glob(osp.join(program_dir, 'ui', '**', '*.py'))
    )
    ui_list = ' '.join(f'"{f}"' for f in sorted(set(ui_files)))

    if args.locale:
        targets = [osp.join(translate_dir, f'{args.locale}.ts')]
    else:
        targets = sorted(glob(osp.join(translate_dir, '*.ts')))

    if not targets:
        raise SystemExit('No .ts files found in translate/.')

    failures = []
    for ts_path in targets:
        rc = _run_update(program_dir, ui_list, ts_path)
        locale = osp.splitext(osp.basename(ts_path))[0]
        if rc == 0:
            print(f'Updated: {ts_path}')
            print(f'Compile with: lrelease translate/{locale}.ts')
        else:
            failures.append((ts_path, rc))

    if failures:
        for ts_path, rc in failures:
            print(f'Failed: {ts_path} (exit={rc})')
        raise SystemExit(1)
