> [!IMPORTANT]  
> **If you're sharing the translated result publicly and no experienced human translator participated in a thorough translation or proofread, please mark it as machine translation somewhere clear to see.**

# BallonsTranslator-Pro
[简体中文](/README_zh_CN.md) | English | [pt-BR](doc/README_PT-BR.md) | [Русский](doc/README_RU.md) | [日本語](doc/README_JA.md) | [Indonesia](doc/README_ID.md) | [Tiếng Việt](doc/README_VI.md) | [한국어](doc/README_KO.md) | [Español](doc/README_ES.md) | [Français](doc/README_FR.md)

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
- Local automation/API documentation for headless and MCP-style workflows (see [Local Automation API](docs/LOCAL_AUTOMATION_API.md))
- Docker/server-mode quickstart plus sample curl/Python client snippets are included in the same document.

---

## Feature map (quick overview)

### Core production pipeline

- **Pipeline modules**: Mix-and-match detector, OCR, translator, inpainter, plus typography/layout passes for scanlation workflows.
- **Layout review & auto-layout**: Layout Review tools, auto-fit/atomic bubble fit, overflow checks, reading-order helpers, and typography QA passes for production lettering.
- **Text editing abilities**: Rich text bubble editor with undo/redo, multi-select transforms, find/replace tools, text style presets, warp/shape controls, and per-block OCR/translation inspection workflows.
- **Quality-first postprocessing**: Includes block-level QA hints, review-oriented tooling, and consistency helpers to reduce obvious translation/lettering regressions.

### Translation quality stack

- **Translation Assist (beta)**: Per-block assist dock with TM/glossary/SFX/concordance candidates, explicit apply, edit-before-apply flow, add-to-TM/glossary actions, block QA warnings, provider telemetry, and cache controls.
- **Compare providers/modules workflow**: Compare candidate outputs by preset (`low_latency`, `high_quality`) and compare scope (`translator`, `ocr`, `detector`, `inpainter`) from one assist surface.
- **Prompt/profile controls**: Per-project assist prompt profiles and provider preference controls are available for controlled quality tuning.
- **Structured quality docs**: Quality guidance and model-ranking references are documented under [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md) and [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md).

### Realtime, automation, and operations

- **Realtime Screen Translator (experimental)**: Project-less live OCR/translation from selected regions (Tools → Realtime Screen Translator), with privacy-first defaults.
- **Automation API**: Local API route discovery (`/routes`) and MCP-style command surface for headless/control-plane tooling, including translation-assist and realtime namespaces.
- **Batch & exports**: Batch queue, archive/CBZ flows, structured OCR/translation interchange, and proof/handoff oriented outputs.
- **Raw downloader**: Registry-based manga/raw source system with explicit legal/safety boundaries and provider extensibility.
- **Diagnostics & troubleshooting**: Environment doctor, startup diagnostics, optional dependency docs, GPU/runtime guidance.

### Platform and usability highlights

- **Windows-first launch ergonomics**: `launcher.bat`/`launch_win.bat` flows for setup + fast launch.
- **In-app Google Fonts installer**: Install and register families without leaving the app.
- **Model/provider compatibility flexibility**: Blend local/offline and cloud-capable OCR/MT/LLM components depending on your constraints.

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

## Docs by use case

### Start here
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- [docs/GPU_ACCELERATION.md](docs/GPU_ACCELERATION.md)
- [docs/DOCS_HIGHLIGHTS.md](docs/DOCS_HIGHLIGHTS.md) — one-page guide describing each major document and when to use it.

### Quality, translation, and lettering
- [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md) — quality/performance expectations for module combinations.
- [docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md) — model/module reference and selection notes.
- [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md) — context/glossary strategy and consistency guidance.
- [docs/INDESIGN_LPTXT_WORKFLOW.md](docs/INDESIGN_LPTXT_WORKFLOW.md) — professional handoff workflow for downstream lettering tools.

### Automation, realtime, and plans
- [docs/LOCAL_AUTOMATION_API.md](docs/LOCAL_AUTOMATION_API.md) — route contract + automation client examples.
- [docs/REALTIME_TRANSLATION_MODE_PLAN.md](docs/REALTIME_TRANSLATION_MODE_PLAN.md) — realtime mode phases and constraints.
- [docs/TRANSLATION_ASSIST_PLAN.md](docs/TRANSLATION_ASSIST_PLAN.md) — translation assist capabilities and roadmap.
- [docs/ALTERNATIVES_FEATURE_GAP_IMPLEMENTATION_PLAN.md](docs/ALTERNATIVES_FEATURE_GAP_IMPLEMENTATION_PLAN.md) — feature-gap plan and checkpoint log.
- [docs/FEATURE_PARITY_MATRIX.md](docs/FEATURE_PARITY_MATRIX.md) — parity status tracker across workflow areas.


## Manga / Comic Raw Downloader

BallonsTranslator-Pro includes a registry-backed manga/raw source downloader. Existing sources (MangaDex, Comick, GOMANGA, Manhwa Reader, MangaForFree, ToonGod, Mangakakalot, NaruRaw, ManhwaRaw, 1kkk, generic chapter URLs, and local folders) are preserved, and new providers can be added through `utils/manga_sources/provider_base.py` plus `utils/manga_sources/registry.py` without manually rebuilding the UI source list.

Current first-batch registry providers include MangaSee/MangaSee123, ReadManga-style pages, and Manganato/Manganelo compatibility. Providers must use public pages/APIs only, respect request delays, and must not bypass DRM, paywalls, login restrictions, private APIs, CAPTCHA, Cloudflare, or other access controls. Browser rendering remains the explicit user-controlled Playwright option.

Developer docs:

- `docs/RAW_SOURCE_PROVIDER_EXPANSION_PLAN.md` — audit and roadmap.
- `docs/MANGA_SOURCE_PLUGIN_API.md` — provider interface, registry metadata, legal/ToS expectations, and tests.
- `docs/EXTERNAL_SOURCE_AUDIT.md` — generated local audit report for optional external downloader repositories.

Smoke tests:

```bash
python scripts/test_manga_sources.py
python scripts/test_manga_sources.py --stable-registry-only
pytest -q tests/test_manga_provider_base.py
```


## Realtime Screen Translator (Experimental)

BallonsTranslator-Pro now includes an optional project-less realtime dialog (Tools → Realtime Screen Translator) for live OCR+translation on a selected screen region. Privacy defaults are local-first: no screenshot/OCR/translation persistence and no live-text logging unless explicitly enabled in future settings.
