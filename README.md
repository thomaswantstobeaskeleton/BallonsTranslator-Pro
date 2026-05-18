> [!IMPORTANT]  
> **If you're sharing the translated result publicly and no experienced human translator participated in a thorough translation or proofread, please mark it as machine translation somewhere clear to see.**

# BallonTranslator
[简体中文](/README.md) | English | [pt-BR](doc/README_PT-BR.md) | [Русский](doc/README_RU.md) | [日本語](doc/README_JA.md) | [Indonesia](doc/README_ID.md) | [Tiếng Việt](doc/README_VI.md) | [한국어](doc/README_KO.md) | [Español](doc/README_ES.md) | [Français](doc/README_FR.md)

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
- Other localized guides:
  - Español: [doc/README_ES.md](doc/README_ES.md)
  - Français: [doc/README_FR.md](doc/README_FR.md)
  - Português (Brasil): [doc/README_PT-BR.md](doc/README_PT-BR.md)
  - Русский: [doc/README_RU.md](doc/README_RU.md)
  - 한국어: [doc/README_KO.md](doc/README_KO.md)
  - 日本語: [doc/README_JA.md](doc/README_JA.md)
  - Bahasa Indonesia: [doc/README_ID.md](doc/README_ID.md)
  - Tiếng Việt: [doc/README_VI.md](doc/README_VI.md)
- Full change history: [docs/CHANGELOG.md](docs/CHANGELOG.md)

---

## Launch the app quickly (what to click)

After cloning/downloading and opening the project folder:

- **Windows (recommended):** double-click `launcher.bat` for a single menu with setup, auto-update, AMD/NVIDIA/CPU GPU modes.
- **Windows quick start:** double-click `launch_win.bat` to start immediately with auto GPU detection.
- **Cross-platform (manual):** run `python launch.py` in terminal.
- **GPU help:** see [docs/GPU_ACCELERATION.md](docs/GPU_ACCELERATION.md), especially for AMD Radeon RX 9070/9060/7900 cards.

If batch files are blocked by Windows SmartScreen, right-click the `.bat` file → **Run as administrator** (or choose **More info → Run anyway**).

---

## Install Google Fonts (native in-app)

You can now install Google Fonts directly from the app:

1. Open **Tools → Models → Install Google Font...**
2. Enter a family name (example: `Bangers`, `Noto Sans JP`, `Comic Neue`)
3. The app downloads and registers the font automatically.

Installed fonts are stored under `fonts/google/` and become available in font pickers.

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
git clone https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro
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

- Python 3.10.2+ (Python 3.10.0/3.10.1 can crash PyInstaller builds with `IndexError: tuple index out of range`)
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
- More automatic bubble lettering with shape-aware safe areas, balanced line breaks, density-aware font scaling, and final overflow safety checks (see [Automatic text formatting](doc/AUTOMATIC_TEXT_FORMATTING.md))
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
