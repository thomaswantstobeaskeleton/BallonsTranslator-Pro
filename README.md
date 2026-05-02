# BallonsTranslator-Pro

BallonsTranslator-Pro is an advanced fork of [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) focused on serious manga/comic translation workflows.

<p align="center">
  <img src="doc/src/1111.png" width="100%">
</p>

At a high level, BallonsTranslator-Pro helps you:

1. Detect speech balloons/text areas
2. Read text (OCR)
3. Translate text
4. Clean original text from artwork (inpaint)
5. Edit and export pages

- 中文文档（与本页同结构）: [README_zh_CN.md](README_zh_CN.md)
- Full change history: [docs/CHANGELOG.md](docs/CHANGELOG.md)

---

## Launch the app quickly (what to click)

After cloning/downloading and opening the project folder:

- **Windows (recommended):** double-click `launch_win.bat`
- **Windows (nightly update + launch):** double-click `launch_win_amd_nightly.bat`
- **Windows (launch + auto-update):** double-click `launch_win_with_autoupdate.bat`
- **Cross-platform (manual):** run `python launch.py` in terminal

If batch files are blocked by Windows SmartScreen, right-click the `.bat` file → **Run as administrator** (or choose **More info → Run anyway**).

---

## Install Google Fonts (native in-app)

You can now install Google Fonts directly from the app:

1. Open **Tools → Models → Install Google Font...**
2. Enter a family name (example: `Bangers`, `Noto Sans JP`, `Comic Neue`)
3. The app downloads and registers the font automatically.

Installed fonts are stored under `fonts/google/` and become available in font pickers.

---

## Build desktop executables (Windows / macOS / Linux)

One-click build scripts are included:

- **Windows:** double-click `build_release.bat`
- **macOS/Linux:** run `bash build_release.sh`

These scripts install packaging dependencies and run PyInstaller using `launch.spec`, outputting app bundles in `dist/`.

---

## Windows installer `.exe` (real app install flow)

To create a Windows installer that behaves like normal software (install folder, Start Menu entry, optional Desktop shortcut):

1. Install **Inno Setup 6** (https://jrsoftware.org/isinfo.php)
2. Run `build_windows_installer.bat`
3. Find the installer in `dist_installer/BallonsTranslatorPro-Setup.exe`

This installer:
- installs into `Program Files`
- registers app uninstall entry in Windows
- adds Start Menu shortcut
- optionally adds Desktop shortcut

---

## Clone / download instructions

### Option A — Download ZIP (easiest)

1. Open the repo page on GitHub.
2. Click **Code** → **Download ZIP**.
3. Extract to a normal folder (example: `Documents/BallonsTranslator-Pro`).
4. Open that folder.
5. Launch using `launch_win.bat` (Windows) or terminal (`python launch.py`).

### Option B — Git clone (terminal)

```bash
git clone <REPO_URL>
cd BallonsTranslator-Pro
python launch.py
```

### Option C — GitHub Desktop

1. Install GitHub Desktop.
2. **File → Clone repository**.
3. Paste repo URL and choose a local path.
4. Open the local folder.
5. Launch `launch_win.bat` (Windows) or `python launch.py`.

---

## Requirements

- Python 3.10+
- Internet for first-time setup/model downloads
- Enough disk space for selected models

Check Python:

```bash
python --version
```

---

## Why Pro

- Practical module combinations for real projects
- Better batch and long-chapter workflow support
- LLM-aided review chain for quality/consistency
- Strong defaults for fast onboarding

---

## Basic workflow

1. **Open pages**: File → Open Folder / Open Images
2. **Select modules**: Detector / OCR / Inpaint / Translator
3. Click **Run**
4. Review and edit text bubbles
5. Export from **Tools → Export all pages**

---

## Useful paths in this repo

- App launcher: `launch.py`
- Windows one-click launcher: `launch_win.bat`
- Config template: `config/config.example.json`
- Active config: `config/config.json`
- Models: `data/models/`
- Fonts: `fonts/`

---

## Docs

- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md)
- [docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md)
- [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md)
- [docs/INDESIGN_LPTXT_WORKFLOW.md](docs/INDESIGN_LPTXT_WORKFLOW.md)
