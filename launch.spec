# PyInstaller spec for BallonsTranslator (launch.py)
import os
import subprocess
from PyInstaller.utils.hooks import collect_data_files

# 获取提交哈希值（如果不是 git 仓库则使用 fallback）
try:
    commit_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        stderr=subprocess.DEVNULL,
    ).decode('utf-8').strip()
except Exception:
    commit_hash = os.environ.get('BATR_COMMIT_HASH', 'nogit')

# 构造带提交哈希值的版本号
version = "1.4.0.dev." + commit_hash

block_cipher = None

base_datas = [
    ('.btrans_cache', './.btrans_cache'),
    ('config', './config'),
    ('data', './data'),
    ('doc', './doc'),
    ('docs', './docs'),
    ('fonts', './fonts'),
    ('icons', './icons'),
    ('modules', './modules'),
    ('scripts', './scripts'),
    ('translate', './translate'),
    ('ui', './ui'),
    ('utils', './utils'),
]

optional_datas = [
    ('venv/lib/python3.12/site-packages/spacy_pkuseg', './spacy_pkuseg'),
    ('venv/lib/python3.12/site-packages/torchvision', './torchvision'),
    ('venv/lib/python3.12/site-packages/translators', './translators'),
    ('venv/lib/python3.12/site-packages/cryptography', './cryptography'),
]

datas = [entry for entry in base_datas if os.path.exists(entry[0])]
datas.extend(entry for entry in optional_datas if os.path.exists(entry[0]))
# Ensure unidic_lite dictionary assets (e.g. dicdir/version) are bundled for manga_ocr tokenizer.
datas.extend(collect_data_files('unidic_lite'))

a = Analysis([
        'launch.py',
        ],
    pathex=[
        './scripts',
        ],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'PyQt6',
        'numpy',
        'urllib3',
        'jaconv',
        'torch',
        'torchvision',
        'transformers',
        'fugashi',
        'unidic_lite',
        'tqdm',
        'shapely',
        'pyclipper',
        'einops',
        'termcolor',
        'bs4',
        'deepl',
        'qtpy',
        'sentencepiece',
        'ctranslate2',
        'docx2txt',
        'piexif',
        'keyboard',
        'requests',
        'colorama',
        'openai',
        'httpx',
        'langdetect',
        'srsly',
        'execjs',
        'pathos',
        ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6",
        "PySide2",
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

win_icon = 'icons/icon2.ico' if os.path.exists('icons/icon2.ico') else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='launch',
    icon=win_icon,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='launch',
)
app = BUNDLE(
    coll,
    name='BallonsTranslator.app',
    icon='icons/icon.icns',
    bundle_identifier=None,
    info_plist={
        'CFBundleDisplayName': 'BallonsTranslator',
        'CFBundleName': 'BallonsTranslator',
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': 'BATR',
        'CFBundleShortVersionString': version,
        'CFBundleVersion': version,
        'CFBundleExecutable': 'launch',
        'CFBundleIconFile': 'icon.icns',
        'CFBundleIdentifier': 'dev.dmmaze.batr',
        'CFBundleInfoDictionaryVersion': '6.0',
        'LSApplicationCategoryType': 'public.app-category.graphics-design',
        'LSEnvironment': {'LANG': 'zh_CN.UTF-8'},
      }
)
