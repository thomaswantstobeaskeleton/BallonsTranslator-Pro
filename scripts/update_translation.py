"""
Update Qt translation .ts file from UI sources.
Run from project root. Requires: pylupdate6 (PyQt6) or pylupdate5 (PyQt5).
Generates translate/<locale>.ts (e.g. zh_CN.ts) from ui/**/*.py.
Compile to .qm with: lrelease translate/zh_CN.ts
"""
import os
import os.path as osp
from glob import glob

from qtpy.QtCore import QLocale
SYSLANG = QLocale.system().name()

if __name__ == '__main__':
    program_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    translate_dir = osp.join(program_dir, 'translate')
    translate_path = osp.join(translate_dir, SYSLANG + '.ts')
    ui_files = (
        glob(osp.join(program_dir, 'ui', '*.py')) +
        glob(osp.join(program_dir, 'ui', '**', '*.py'))
    )
    ui_list = ' '.join(f'"{f}"' for f in sorted(set(ui_files)))
    cmd = f'pylupdate6 -verbose {ui_list} -ts "{translate_path}"'
    try:
        import subprocess
        subprocess.run(cmd, shell=True, cwd=program_dir)
    except Exception:
        cmd5 = f'pylupdate5 -verbose {ui_list} -ts "{translate_path}"'
        os.system(cmd5)
    print(f'target language: {SYSLANG}')
    print(f'Saved to {translate_path}')
    print('Compile with: lrelease translate/' + SYSLANG + '.ts')